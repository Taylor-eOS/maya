import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import soundfile as sf
import numpy as np

TEMPERATURE = 0.1
TOP_P = 1.0
REPETITION_PENALTY = 1.0
CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7
SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
BOS_ID = 128000
TEXT_EOT_ID = 128009

class AudioGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            "maya-research/maya1", 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "maya-research/maya1", 
            trust_remote_code=True
        )
        try:
            self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
            if torch.cuda.is_available():
                self.snac_model = self.snac_model.to(self.device)
        except Exception as e:
            print(f"Error loading SNAC model: {e}")
            raise

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
        snac_codes = [token_id for token_id in token_ids[:eos_idx] if SNAC_MIN_ID <= token_id <= SNAC_MAX_ID]
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
            slots = snac_tokens[i*7:(i+1)*7]
            l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
            l2.extend([(slots[1] - CODE_TOKEN_OFFSET) % 4096, (slots[4] - CODE_TOKEN_OFFSET) % 4096])
            l3.extend([
                (slots[2] - CODE_TOKEN_OFFSET) % 4096,
                (slots[3] - CODE_TOKEN_OFFSET) % 4096,
                (slots[5] - CODE_TOKEN_OFFSET) % 4096,
                (slots[6] - CODE_TOKEN_OFFSET) % 4096,
            ])
        return [l1, l2, l3]

    def generate_audio(self, text, speaker_description, max_new_tokens=16384):
        prompt = self.build_prompt(speaker_description, text)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]
        word_count = len(text.split())
        estimated_tokens = int(1.5 * 30 * word_count + 128)
        safe_max = min(estimated_tokens, max_new_tokens, 8192, 131072 - input_len - 100)
        print(f"Generating audio for {len(text)} characters, {word_count} words")
        print(f"Input tokens: {input_len}, Estimated: {estimated_tokens}, Max new tokens: {safe_max}")
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=safe_max,
                min_new_tokens=28,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                do_sample=True,
                eos_token_id=CODE_END_TOKEN_ID,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated_ids = outputs[0, input_len:].tolist()
        snac_tokens = self.extract_snac_codes(generated_ids)
        print(f"Generated {len(snac_tokens)} SNAC tokens")
        if len(snac_tokens) < 7:
            raise ValueError(f"Not enough SNAC tokens generated for audio: {len(snac_tokens)}")
        levels = self.unpack_snac_from_7(snac_tokens)
        codes_tensor = [torch.tensor(level, dtype=torch.long, device=self.device).unsqueeze(0) for level in levels]
        with torch.inference_mode():
            z_q = self.snac_model.quantizer.from_codes(codes_tensor)
            audio = self.snac_model.decoder(z_q)[0, 0].cpu().numpy()
        audio_abs = np.abs(audio)
        threshold = np.max(audio_abs) * 0.01
        if len(audio) > 2048:
            start_idx = np.where(audio_abs > threshold)[0]
            if len(start_idx) > 0:
                start_idx = max(0, start_idx[0] - 512)
                audio = audio[start_idx:]
            else:
                audio = audio[2048:]
        end_idx = np.where(audio_abs > threshold)[0]
        if len(end_idx) > 0:
            end_idx = min(len(audio) - 1, end_idx[-1] + 512)
            audio = audio[:end_idx + 1]
        print(f"Generated audio length: {len(audio)} samples ({len(audio)/24000:.2f} seconds)")
        return audio

    def save_audio(self, audio, output_file, sample_rate=24000):
        sf.write(output_file, audio, sample_rate)

if __name__ == "__main__":
    generator = AudioGenerator()
    text = "Frodo and Sam walk to Mordor to return the ring."
    speaker_description = "Professional british male audiobook narrator."
    audio = generator.generate_audio(text, speaker_description)
    generator.save_audio(audio, "output.wav")
    print("Audio saved to output.wav")

