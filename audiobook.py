import re
import os
import torch
from pathlib import Path
import soundfile as sf
from pydub import AudioSegment
import pysbd
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC

input_file = "input.txt"
output_dir = "."
test_mode_on = False
log_file = "log.txt"
max_sentences_per_audio_file = 120
speaker_description = "Professional audiobook narrator, a native German that speaks good english with no accent, calm and pleasant, clear pronounciation."

SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
CODE_START_TOKEN_ID = 128257
TEXT_EOT_ID = 128009
CODE_END_TOKEN_ID = 128258
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
CODE_TOKEN_OFFSET = 128266
SNAC_TOKENS_PER_FRAME = 7

class AudioGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained("maya-research/maya1", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("maya-research/maya1", trust_remote_code=True)
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        if torch.cuda.is_available():
            self.snac_model = self.snac_model.to(self.device)

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

    def generate_audio(self, text, speaker_description):
        prompt = self.build_prompt(speaker_description, text)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                min_new_tokens=28,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                eos_token_id=CODE_END_TOKEN_ID,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated_ids = outputs[0, inputs['input_ids'].shape[1]:].tolist()
        snac_tokens = self.extract_snac_codes(generated_ids)
        if len(snac_tokens) < 7:
            raise ValueError("Not enough SNAC tokens generated for audio")
        levels = self.unpack_snac_from_7(snac_tokens)
        codes_tensor = [torch.tensor(level, dtype=torch.long, device=self.device).unsqueeze(0) for level in levels]
        with torch.inference_mode():
            z_q = self.snac_model.quantizer.from_codes(codes_tensor)
            audio = self.snac_model.decoder(z_q)[0, 0].cpu().numpy()
        if len(audio) > 2048:
            audio = audio[2048:]
        else:
            audio = audio[2048:len(audio)]
        return audio

    def save_audio(self, audio, output_file, sample_rate=24000):
        sf.write(output_file, audio, sample_rate)

def split_into_chapters(text):
    parts = re.split(r'\n\s*\n+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def split_chapter_sentences(chap, max_sentences=max_sentences_per_audio_file):
    paragraph_placeholder = "||PARAGRAPH||"
    paragraph_breaks = re.split(r'(\n{2,})', chap)
    segmenter = pysbd.Segmenter(language="en", clean=True)
    elements = []
    for part in paragraph_breaks:
        if not part:
            continue
        if re.match(r'\n{2,}', part):
            elements.append(paragraph_placeholder)
        else:
            elements.extend(segmenter.segment(part.strip()))
    processed = []
    for el in elements:
        if el == paragraph_placeholder:
            processed.append("\n\n")
        else:
            processed.append(el + "\n\n")
    chunks = [processed[i:i+max_sentences] for i in range(0, len(processed), max_sentences)]
    if len(chunks) > 1:
        a, b = chunks[-2], chunks[-1]
        combined = a + b
        half = len(combined) // 2
        chunks[-2], chunks[-1] = combined[:half], combined[half:]
    return ["".join(chunk).strip() for chunk in chunks]

def main():
    text = Path(input_file).read_text(encoding='utf-8')
    chapters = split_into_chapters(text)
    idx = 1
    generator = AudioGenerator()
    for chap in chapters:
        for chunk in split_chapter_sentences(chap):
            if test_mode_on:
                with open(log_file, "a") as f:
                    f.write(f"Chunk {idx}: {chunk}\n\n")
            else:
                wav_path = os.path.join(output_dir, f"chunk_{idx}.wav")
                audio = generator.generate_audio(chunk, speaker_description)
                generator.save_audio(audio, wav_path)
                mp3_path = os.path.join(output_dir, f"chunk_{idx}.mp3")
                AudioSegment.from_wav(wav_path).export(mp3_path, format='mp3')
                try:
                    os.remove(wav_path)
                except FileNotFoundError:
                    print(f"Error removing wav file: {wav_path}")
            idx += 1

if __name__ == "__main__":
    main()

