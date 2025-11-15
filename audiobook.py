import os
import torch
import numpy as np
from pysbd import Segmenter
from maya import AudioGenerator
import soundfile as sf

def split_text_into_chunks(text, max_words=100, min_tail_ratio=0.5, segmenter=None):
    if not text or text.strip() == "":
        return []
    if segmenter is None:
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    else:
        sentences = segmenter.segment(text)
    sent_words = [len(s.split()) for s in sentences]
    chunks = []
    curr = []
    curr_words = 0
    for s, w in zip(sentences, sent_words):
        if w > max_words:
            words = s.split()
            i = 0
            while i < len(words):
                take = min(max_words, len(words) - i)
                chunk = " ".join(words[i:i+take])
                chunks.append(chunk)
                i += take
            continue
        if curr_words + w > max_words:
            if curr:
                chunks.append(" ".join(curr))
            curr = [s]
            curr_words = w
        else:
            curr.append(s)
            curr_words += w
    if curr:
        chunks.append(" ".join(curr))
    if len(chunks) >= 2:
        last_words = len(chunks[-1].split())
        prev_words = len(chunks[-2].split())
        can_merge_without_exceed = (prev_words + last_words) <= max_words
        tail_too_small = last_words < max(1, int(max_words * min_tail_ratio))
        if tail_too_small and can_merge_without_exceed:
            chunks[-2] = chunks[-2] + " " + chunks[-1]
            chunks.pop(-1)
    return chunks

gen = AudioGenerator()
segmenter = Segmenter(language="en", clean=False)
speaker_description = "Realistic male audiobook narrator in the 50s age with british accent. Normal pitch, warm timbre, friendly tone, reading pacing, good pronounciation."
fixed_seed = 42
max_new_tokens = 16384
sample_rate = 24000
output_dir = "."
os.makedirs(output_dir, exist_ok=True)

with open("input.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

sections = [section.strip() for section in full_text.split("\n\n") if section.strip()]

for section_index, section in enumerate(sections):
    print(f"Processing section {section_index + 1}/{len(sections)}")
    full_text_for_check = " ".join(section.splitlines())
    word_count = len(full_text_for_check.split())
    print(f"Section has {word_count} words")
    chunks = split_text_into_chunks(full_text_for_check, max_words=100, min_tail_ratio=0.5, segmenter=segmenter)
    print(f"Split into {len(chunks)} chunks")
    audio_segments = []
    for idx, chunk in enumerate(chunks):
        print(f"Generating chunk {idx+1}/{len(chunks)}: {len(chunk.split())} words")
        torch.manual_seed(fixed_seed)
        try:
            audio = gen.generate_audio(chunk, speaker_description, max_new_tokens)
            audio_segments.append(audio)
        except Exception as e:
            print(f"Error generating audio for chunk {idx+1}: {e}")
            words = chunk.split()
            if len(words) > 90:
                fallback_chunk = " ".join(words[:90])
                try:
                    torch.manual_seed(fixed_seed)
                    audio = gen.generate_audio(fallback_chunk, speaker_description, max_new_tokens)
                    audio_segments.append(audio)
                    print(f"Fallback succeeded for chunk {idx+1}")
                except Exception as e2:
                    print(f"Fallback failed for chunk {idx+1}: {e2}")
            else:
                print(f"No fallback available for chunk {idx+1}")
    if audio_segments:
        full_audio = np.concatenate(audio_segments)
        output_file = os.path.join(output_dir, f"output_{section_index + 1}.wav")
        sf.write(output_file, full_audio, sample_rate)
        print(f"Saved audio for section {section_index + 1} to {output_file}")
        print(f"Generated {len(audio_segments)} audio segments")
    else:
        print(f"No audio generated for section {section_index + 1}")

