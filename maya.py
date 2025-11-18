import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from snac import SNAC
import soundfile as sf
import numpy as np
import pysbd
from ids import CODE_START_TOKEN_ID, CODE_END_TOKEN_ID, CODE_TOKEN_OFFSET, SNAC_MIN_ID, SNAC_MAX_ID, SOH_ID, EOH_ID, SOA_ID, TEXT_EOT_ID

TEMPERATURE = 0.7
TOP_P = 0.95
REPETITION_PENALTY = 1.15
SNAC_TOKENS_PER_FRAME = 7

class CallbackStoppingCriteria(StoppingCriteria):
    def __init__(self, callback):
        self.callback = callback
    def __call__(self, input_ids, scores, **kwargs):
        last_token = input_ids[0, -1].item()
        return self.callback(last_token)

class MaxTokensStoppingCriteria(StoppingCriteria):
    def __init__(self, max_total_tokens):
        self.max_total_tokens = max_total_tokens
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids.shape[1] >= self.max_total_tokens

class AudioGenerator:
    def __init__(self, model_name="maya-research/maya1", snac_name="hubertsiuzdak/snac_24khz"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.snac_model = SNAC.from_pretrained(snac_name).eval()
        if torch.cuda.is_available():
            self.snac_model = self.snac_model.to(self.device)
        self.segmenter = pysbd.Segmenter(language="en", clean=False)

    def build_prompt(self, description, text):
        soh_token = self.tokenizer.decode([SOH_ID])
        eoh_token = self.tokenizer.decode([EOH_ID])
        soa_token = self.tokenizer.decode([SOA_ID])
        sos_token = self.tokenizer.decode([CODE_START_TOKEN_ID])
        eot_token = self.tokenizer.decode([TEXT_EOT_ID])
        bos_token = self.tokenizer.bos_token
        formatted_text = f'<description="{description}"> {text}'
        prompt = soh_token + bos_token + formatted_text + eot_token + eoh_token + soa_token + sos_token
        return prompt

    def extract_snac_codes(self, token_ids):
        try:
            eos_idx = token_ids.index(CODE_END_TOKEN_ID)
        except ValueError:
            eos_idx = len(token_ids)
        snac_codes = [tid for tid in token_ids[:eos_idx] if SNAC_MIN_ID <= tid <= SNAC_MAX_ID]
        return snac_codes

    def unpack_snac_from_7(self, snac_tokens):
        if not snac_tokens:
            return [[], [], []]
        if snac_tokens and snac_tokens[-1] == CODE_END_TOKEN_ID:
            snac_tokens = snac_tokens[:-1]
        frames = len(snac_tokens) // SNAC_TOKENS_PER_FRAME
        snac_tokens = snac_tokens[:frames * SNAC_TOKENS_PER_FRAME]
        if frames == 0:
            return [[], [], []]
        l1, l2, l3 = [], [], []
        for i in range(frames):
            slots = snac_tokens[i * 7:(i + 1) * 7]
            l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
            l2.extend([(slots[1] - CODE_TOKEN_OFFSET) % 4096, (slots[4] - CODE_TOKEN_OFFSET) % 4096])
            l3.extend([
                (slots[2] - CODE_TOKEN_OFFSET) % 4096,
                (slots[3] - CODE_TOKEN_OFFSET) % 4096,
                (slots[5] - CODE_TOKEN_OFFSET) % 4096,
                (slots[6] - CODE_TOKEN_OFFSET) % 4096,
            ])
        return [l1, l2, l3]

    def generate_audio(self, text, speaker_description, temperature=TEMPERATURE, top_p=TOP_P, repetition_penalty=REPETITION_PENALTY):
        prompt = self.build_prompt(speaker_description, text)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        max_audio_tokens = 16384
        consecutive_silence_frames = 0
        generated_tokens = []
        def stop_callback(new_token_id):
            nonlocal consecutive_silence_frames, generated_tokens
            generated_tokens.append(new_token_id)
            if new_token_id == CODE_END_TOKEN_ID:
                return True
            if not (SNAC_MIN_ID <= new_token_id <= SNAC_MAX_ID):
                return False
            if len(generated_tokens) % SNAC_TOKENS_PER_FRAME != 0:
                return False
            coarse_code = (generated_tokens[-SNAC_TOKENS_PER_FRAME] - CODE_TOKEN_OFFSET) % 4096
            if coarse_code >= 2048:
                consecutive_silence_frames += 1
            else:
                consecutive_silence_frames = 0
            if consecutive_silence_frames >= 10:
                return True
            return False
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_audio_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                eos_token_id=CODE_END_TOKEN_ID,
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([
                    CallbackStoppingCriteria(stop_callback),
                    MaxTokensStoppingCriteria(input_len + max_audio_tokens)
                ])
            )
        generated_ids = outputs[0, input_len:].tolist()
        snac_tokens = self.extract_snac_codes(generated_ids)
        if len(snac_tokens) < SNAC_TOKENS_PER_FRAME:
            raise ValueError("Model generated almost no audio codes")
        levels = self.unpack_snac_from_7(snac_tokens)
        codes_tensor = [torch.tensor(level, dtype=torch.long, device=self.device).unsqueeze(0) for level in levels]
        with torch.inference_mode():
            z_q = self.snac_model.quantizer.from_codes(codes_tensor)
            audio = self.snac_model.decoder(z_q)[0, 0].cpu().numpy()
        audio_abs = np.abs(audio)
        threshold = np.max(audio_abs) * 0.01 if audio_abs.size > 0 else 0
        if len(audio) > 2048:
            start_idx = np.where(audio_abs[2048:] > threshold)[0]
            if len(start_idx) > 0:
                start_idx = 2048 + start_idx[0] - 512
                start_idx = max(0, start_idx)
                audio = audio[start_idx:]
        end_idx = np.where(audio_abs > threshold)[0]
        if len(end_idx) > 0:
            end_idx = end_idx[-1] + 1024
            audio = audio[:end_idx + 1]
        return audio, snac_tokens

    def save_audio(self, audio, output_file, sample_rate=24000):
        sf.write(output_file, audio, sample_rate)

if __name__ == "__main__":
    generator = AudioGenerator()
    text = "Frodo and Sam walk to Mordor to return the ring."
    speaker_description = "Professional british male audiobook narrator."
    audio, tokens = generator.generate_audio(text, speaker_description)
    generator.save_audio(audio[0], "output.wav")
    print("Audio saved to output.wav")
