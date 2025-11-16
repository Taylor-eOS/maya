import os
import torch
import numpy as np
from pysbd import Segmenter
from maya import AudioGenerator
import soundfile as sf

speaker_description = "A male audiobook narrator in a warm, mid-range voice with a subtle continental European lilt, that sounds like he is reading from a history book, maintaining smooth pacing and pausing only after full thoughts for natural flow."
fixed_seed = 42
pause_duration = 0.4
max_new_tokens = 8192
sample_rate = 24000
pause_length = int(pause_duration * sample_rate)
pause_silence = np.zeros(pause_length, dtype=np.float32)
output_dir = "."
input_file = "input.txt"
max_words_per_chunk = 80
os.makedirs(output_dir, exist_ok=True)
gen = AudioGenerator()
segmenter = Segmenter(language="en", clean=False)

def split_text_into_chunks(text, max_words=80, segmenter=None):
    sentences = segmenter.segment(text)
    chunks = []
    for s in sentences:
        s = s.strip()
        w = len(s.split())
        if w <= max_words:
            chunks.append(s)
        else:
            words = s.split()
            i = 0
            while i < len(words):
                take = min(max_words, len(words) - i)
                sub = " ".join(words[i:i+take])
                chunks.append(sub)
                i += take
    return chunks

def main():
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
            print(f"Generating chunk {idx+1}/{len(chunks)}: {len(chunk.split())} words")
            torch.manual_seed(fixed_seed)
            try:
                audio = gen.generate_audio(chunk, speaker_description, max_new_tokens)
                audio_segments.append(audio)
            except Exception as e:
                print(f"Error generating audio for chunk {idx+1}: {e}")
        if audio_segments:
            paused_segments = [pause_silence, audio_segments[0]]
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

