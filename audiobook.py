import os
import torch
import numpy as np
import soundfile as sf
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
from pysbd import Segmenter

SPEAKER = "A British male in the 40s reading a history book without accent professionally, pronouncing clearly and holding pauses only at natural logical breaks."
MAX_WORDS_PER_CHUNK = 80
TEMPERATURE = 0.2
TOP_P = 0.95
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
DEFAULT_WPM = 140
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 2048
PAUSE_DURATION = 0.4
FIXED_SEED = 42
MAX_NEW_TOKENS = 8192
FADE_DURATION = 0.05
OUTPUT_DIR = "."
INPUT_FILE = "input.txt"
VERBOSE = False

class AudioGenerator:
    def __init__(self, device=None, verbose=True):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        if self.verbose:
            print(f"Using device: {self.device}")
        self.model = AutoModelForCausalLM.from_pretrained("maya-research/maya1", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("maya-research/maya1", trust_remote_code=True)
        try:
            self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
            if torch.cuda.is_available():
                self.snac_model = self.snac_model.to(self.device)
        except Exception as e:
            print("Error loading SNAC model:", e)
            raise

    def build_prompt(self, description, text):
        soh_token = self.tokenizer.decode([SOH_ID])
        eoh_token = self.tokenizer.decode([EOH_ID])
        soa_token = self.tokenizer.decode([SOA_ID])
        sos_token = self.tokenizer.decode([CODE_START_TOKEN_ID])
        eot_token = self.tokenizer.decode([TEXT_EOT_ID])
        bos_token = self.tokenizer.bos_token or ""
        formatted_text = f'<description="{description}"> {text}'
        return soh_token + bos_token + formatted_text + eot_token + eoh_token + soa_token + sos_token

    def extract_snac_codes(self, token_ids):
        try:
            eos_idx = token_ids.index(CODE_END_TOKEN_ID)
        except ValueError:
            eos_idx = len(token_ids)
        snac_region = token_ids[:eos_idx]
        snac_codes = [tid for tid in snac_region if SNAC_MIN_ID <= tid <= SNAC_MAX_ID]
        if self.verbose:
            print("SNAC extraction diagnostics:", {
                "full_generated_len": len(token_ids),
                "region_len": len(snac_region),
                "snac_count": len(snac_codes),
                "preview_region": snac_region[:120],
                "preview_snac": snac_codes[:120]
            })
        return snac_codes

    def unpack_snac(self, snac_tokens, tokens_per_frame=SNAC_TOKENS_PER_FRAME):
        if not snac_tokens:
            return [[], [], []]
        try:
            cfg = getattr(self.snac_model, "config", None)
            if cfg is not None:
                if hasattr(cfg, "tokens_per_frame"):
                    tokens_per_frame = cfg.tokens_per_frame
                if hasattr(cfg, "code_offset"):
                    global CODE_TOKEN_OFFSET
                    CODE_TOKEN_OFFSET = cfg.code_offset
        except Exception:
            pass
        total = len(snac_tokens)
        frames = total // tokens_per_frame
        remainder = total % tokens_per_frame
        if self.verbose:
            print(f"Tokens per frame used: {tokens_per_frame}, full frames: {frames}, remainder: {remainder}")
        if frames == 0 and remainder == 0:
            return [[], [], []]
        l1 = []
        l2 = []
        l3 = []
        for i in range(frames):
            slots = snac_tokens[i * tokens_per_frame:(i + 1) * tokens_per_frame]
            if tokens_per_frame == 7:
                l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
                l2.extend([(slots[1] - CODE_TOKEN_OFFSET) % 4096, (slots[4] - CODE_TOKEN_OFFSET) % 4096])
                l3.extend([(slots[2] - CODE_TOKEN_OFFSET) % 4096, (slots[3] - CODE_TOKEN_OFFSET) % 4096, (slots[5] - CODE_TOKEN_OFFSET) % 4096, (slots[6] - CODE_TOKEN_OFFSET) % 4096])
            else:
                l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
                if len(slots) >= 3:
                    l2.extend([(slots[1] - CODE_TOKEN_OFFSET) % 4096, (slots[2] - CODE_TOKEN_OFFSET) % 4096])
                for s in slots[3:]:
                    l3.append((s - CODE_TOKEN_OFFSET) % 4096)
        if remainder > 0:
            slots = snac_tokens[frames * tokens_per_frame:]
            slots.extend([CODE_TOKEN_OFFSET] * (tokens_per_frame - len(slots)))
            if self.verbose:
                print(f"Padding remainder {remainder} tokens to full frame with silence")
            if tokens_per_frame == 7:
                l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
                l2.extend([(slots[1] - CODE_TOKEN_OFFSET) % 4096, (slots[4] - CODE_TOKEN_OFFSET) % 4096])
                l3.extend([(slots[2] - CODE_TOKEN_OFFSET) % 4096, (slots[3] - CODE_TOKEN_OFFSET) % 4096, (slots[5] - CODE_TOKEN_OFFSET) % 4096, (slots[6] - CODE_TOKEN_OFFSET) % 4096])
            else:
                l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
                if len(slots) >= 3:
                    l2.extend([(slots[1] - CODE_TOKEN_OFFSET) % 4096, (slots[2] - CODE_TOKEN_OFFSET) % 4096])
                for s in slots[3:]:
                    l3.append((s - CODE_TOKEN_OFFSET) % 4096)
        if self.verbose:
            print(f"Level lengths: l1={len(l1)}, l2={len(l2)}, l3={len(l3)}")
        return [l1, l2, l3]

    def trim_leading_silence_rms(self, audio, sr=SAMPLE_RATE, win_ms=10, threshold_db=-45.0, prepad=0.02):
        win = int(sr * (win_ms / 1000.0))
        if win < 1:
            win = 1
        pad = int(sr * prepad)
        if len(audio) <= win:
            return audio[pad:]
        try:
            framed = np.lib.stride_tricks.sliding_window_view(np.abs(audio), win)
            rms = np.sqrt(np.mean(framed ** 2, axis=1))
            rms_db = 20 * np.log10(np.maximum(rms, 1e-12))
            idx = np.where(rms_db > threshold_db)[0]
            if idx.size == 0:
                return audio[pad:]
            first = max(0, idx[0] - pad)
            return audio[first:]
        except Exception:
            abs_thresh = np.max(np.abs(audio)) * 0.01
            start_idx = np.where(np.abs(audio) > abs_thresh)[0]
            if start_idx.size == 0:
                return audio[pad:]
            start_idx = max(0, start_idx[0] - pad)
            return audio[start_idx:]

    def trim_trailing_silence_rms(self, audio, sr=SAMPLE_RATE, win_ms=10, threshold_db=-45.0, postpad=0.02):
        win = int(sr * (win_ms / 1000.0))
        if win < 1:
            win = 1
        pad = int(sr * postpad)
        if len(audio) <= win:
            return audio[:-pad] if len(audio) > pad else audio
        try:
            framed = np.lib.stride_tricks.sliding_window_view(np.abs(audio), win)
            rms = np.sqrt(np.mean(framed ** 2, axis=1))
            rms_db = 20 * np.log10(np.maximum(rms, 1e-12))
            idx = np.where(rms_db > threshold_db)[0]
            if idx.size == 0:
                return audio[:-pad] if len(audio) > pad else audio
            last = min(len(rms_db) - 1, idx[-1] + pad // max(1, win))
            end_idx = min(len(audio), (last * 1) + win)
            return audio[:end_idx]
        except Exception:
            return audio

    def generate_audio(self, text, speaker_description, max_new_tokens=16384, wpm=DEFAULT_WPM, seconds_per_word_override=None):
        start_time = time.time()
        word_count = len(text.split())
        if seconds_per_word_override is None:
            seconds_per_word = 60.0 / float(max(1, wpm))
        else:
            seconds_per_word = seconds_per_word_override
        estimated_tokens = int(1.2 * 30 * word_count + 128)
        prompt = self.build_prompt(speaker_description, text)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]
        safe_max = min(estimated_tokens, max_new_tokens, 8192, 131072 - input_len - 100)
        min_new = int(15 * word_count)
        if self.verbose:
            print(f"Generating audio for {len(text)} characters, {word_count} words")
            print(f"Input tokens: {input_len}, Estimated tokens: {estimated_tokens}, Max new tokens: {safe_max}, Min new tokens: {min_new}")
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=safe_max, min_new_tokens=min_new, temperature=TEMPERATURE, top_p=TOP_P, repetition_penalty=REPETITION_PENALTY, do_sample=True, eos_token_id=CODE_END_TOKEN_ID, pad_token_id=self.tokenizer.pad_token_id)
        generated_ids = outputs[0, input_len:].tolist()
        if self.verbose:
            print(f"Generated raw ids length: {len(generated_ids)}. First 160 ids: {generated_ids[:160]}")
        snac_tokens = self.extract_snac_codes(generated_ids)
        if self.verbose:
            print(f"Generated {len(snac_tokens)} SNAC tokens")
        if len(snac_tokens) == 0:
            raise ValueError(f"No SNAC tokens generated for audio despite minimum")
        tokens_per_frame = SNAC_TOKENS_PER_FRAME
        try:
            cfg = getattr(self.snac_model, "config", None)
            if cfg is not None and hasattr(cfg, "tokens_per_frame"):
                tokens_per_frame = cfg.tokens_per_frame
        except Exception:
            pass
        max_seconds = max(1.0, word_count * seconds_per_word)
        max_frames = max(1, int((max_seconds * SAMPLE_RATE) / SAMPLES_PER_FRAME))
        possible_frames = (len(snac_tokens) + tokens_per_frame - 1) // tokens_per_frame
        if possible_frames > max_frames:
            truncate_tokens = max_frames * tokens_per_frame
            if self.verbose:
                print(f"Truncating SNAC tokens to {truncate_tokens} ({max_frames} frames) to limit decoded duration ~{max_seconds:.1f}s")
            snac_tokens = snac_tokens[:truncate_tokens]
        levels = self.unpack_snac(snac_tokens, tokens_per_frame=tokens_per_frame)
        codes_tensor = [torch.tensor(level, dtype=torch.long, device=self.device).unsqueeze(0) for level in levels]
        if self.verbose:
            for i, t in enumerate(codes_tensor):
                try:
                    print(f"codes_tensor[{i}].shape dtype:{t.dtype} min/max:{int(t.min())}/{int(t.max())} len:{t.shape}")
                except Exception:
                    print(f"codes_tensor[{i}] inspection failed")
        codes_tensor = [ct.long().to(self.device) for ct in codes_tensor]
        with torch.inference_mode():
            z_q = self.snac_model.quantizer.from_codes(codes_tensor)
            audio = self.snac_model.decoder(z_q)[0, 0].cpu().numpy()
        audio = self.trim_leading_silence_rms(audio, sr=SAMPLE_RATE, win_ms=10, threshold_db=-45.0, prepad=0.02)
        audio = self.trim_trailing_silence_rms(audio, sr=SAMPLE_RATE, win_ms=10, threshold_db=-45.0, postpad=0.02)
        if len(audio) > 2048:
            audio_abs = np.abs(audio)
            threshold = np.max(audio_abs) * 0.01
            start_idx = np.where(audio_abs > threshold)[0]
            if len(start_idx) > 0:
                start_idx = max(0, start_idx[0] - 512)
                audio = audio[start_idx:]
            else:
                audio = audio[2048:]
        elapsed_time = time.time() - start_time
        time_per_word = elapsed_time / word_count if word_count > 0 else 0
        if self.verbose:
            print(f"Generated audio length: {len(audio)} samples ({len(audio) / SAMPLE_RATE:.2f} seconds)")
        print(f"Generation time: {elapsed_time:.2f}s for {word_count} words ({time_per_word:.3f}s per word)")
        return audio

    def save_audio(self, audio, output_file, sample_rate=SAMPLE_RATE):
        sf.write(output_file, audio, sample_rate)

def split_text_into_chunks(text, max_words=80, segmenter=None):
    sentences = segmenter.segment(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    processed_sentences = []
    for s in sentences:
        words = s.split()
        if len(words) > max_words:
            for k in range(0, len(words), max_words):
                sub = ' '.join(words[k:k + max_words])
                processed_sentences.append(sub)
        else:
            processed_sentences.append(s)
    sentences = processed_sentences
    word_counts = [len(s.split()) for s in sentences]
    if not sentences:
        return []
    n = len(sentences)
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    parent = [-1] * (n + 1)
    variance = [0] * (n + 1)
    for i in range(n):
        if word_counts[i] > max_words:
            j = i
            current_words = word_counts[i]
            chunk_count = dp[i] + 1
            future_variance = variance[i] + (current_words - max_words / 2.0) ** 2
            if chunk_count < dp[j + 1] or (chunk_count == dp[j + 1] and future_variance < variance[j + 1]):
                dp[j + 1] = chunk_count
                parent[j + 1] = i
                variance[j + 1] = future_variance
            continue
        current_words = 0
        for j in range(i, n):
            current_words += word_counts[j]
            if current_words > max_words:
                break
            chunk_count = dp[i] + 1
            future_variance = variance[i] + (current_words - max_words / 2.0) ** 2
            if chunk_count < dp[j + 1] or (chunk_count == dp[j + 1] and future_variance < variance[j + 1]):
                dp[j + 1] = chunk_count
                parent[j + 1] = i
                variance[j + 1] = future_variance
    chunks = []
    idx = n
    while idx > 0:
        start = parent[idx]
        chunk_sentences = sentences[start:idx]
        chunks.append(" ".join(chunk_sentences))
        idx = start
    chunks.reverse()
    return chunks

def apply_fade(audio, fade_length):
    if len(audio) <= 2 * fade_length:
        return audio
    fade_out = np.linspace(1.0, 0.0, fade_length)
    fade_in = np.linspace(0.0, 1.0, fade_length)
    faded = audio.copy()
    faded[:fade_length] *= fade_in
    faded[-fade_length:] *= fade_out
    return faded

def main():
    speaker_description = SPEAKER
    pause_duration = PAUSE_DURATION
    fixed_seed = FIXED_SEED
    max_new_tokens = MAX_NEW_TOKENS
    sample_rate = SAMPLE_RATE
    fade_duration = FADE_DURATION
    output_dir = OUTPUT_DIR
    input_file = INPUT_FILE
    max_words_per_chunk = MAX_WORDS_PER_CHUNK
    verbose = VERBOSE
    pause_length = int(pause_duration * sample_rate)
    pause_silence = np.zeros(pause_length, dtype=np.float32)
    fade_length = int(fade_duration * sample_rate)
    os.makedirs(output_dir, exist_ok=True)
    gen = AudioGenerator(verbose=verbose)
    segmenter = Segmenter(language="en", clean=False)
    with open(input_file, "r", encoding="utf-8") as f:
        full_text = f.read()
    sections = [section.strip() for section in full_text.split("\n\n") if section.strip()]
    for section_index, section in enumerate(sections):
        print(f"Processing section {section_index + 1}/{len(sections)}")
        full_text_for_check = " ".join(section.splitlines())
        word_count = len(full_text_for_check.split())
        print(f"Section has {word_count} words")
        chunks = split_text_into_chunks(full_text_for_check, max_words=max_words_per_chunk, segmenter=segmenter)
        print(f"Split into {len(chunks)} chunks")
        audio_segments = []
        for idx, chunk in enumerate(chunks):
            print(f"Generating chunk {idx + 1}/{len(chunks)}: {len(chunk.split())} words")
            torch.manual_seed(fixed_seed)
            try:
                audio = gen.generate_audio(chunk, speaker_description, max_new_tokens=max_new_tokens, wpm=DEFAULT_WPM)
                if len(audio) == 0:
                    print(f"Warning: Chunk {idx + 1} produced no audio, skipping")
                    continue
                faded_audio = apply_fade(audio, fade_length)
                audio_segments.append(faded_audio)
            except Exception as e:
                print(f"Error generating audio for chunk {idx + 1}: {e}")
        if audio_segments:
            paused_segments = [audio_segments[0]]
            for seg in audio_segments[1:]:
                paused_segments.append(pause_silence)
                paused_segments.append(seg)
            full_audio = np.concatenate(paused_segments)
            output_file = os.path.join(output_dir, f"output_{section_index + 1}.wav")
            sf.write(output_file, full_audio, sample_rate)
            print(f"Saved audio for section {section_index + 1} to {output_file}")
            print(f"Generated {len(audio_segments)} audio segments")
        else:
            print(f"No audio generated for section {section_index + 1}")

if __name__ == "__main__":
    main()

