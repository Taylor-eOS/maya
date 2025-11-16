import os
import torch
import numpy as np
from pysbd import Segmenter
from maya import AudioGenerator
import soundfile as sf

speaker_description = "A steady male voice reading a history audiobook, pitched in the comfortable mid range around 150 hertz, with a gentle warmth from soft rounded vowels and even breath support, carrying a faint neutral European inflection like a calm professor from Copenhagen, speaking at a measured 140 words per minute, rising slightly for emphasis on key ideas but never rushing, and holding brief natural pauses only at the end of complete sentences to let thoughts settle before the next, content and happy to share."
fixed_seed = 42
pause_duration = 0.4
max_new_tokens = 8192
sample_rate = 24000
pause_length = int(pause_duration * sample_rate)
pause_silence = np.zeros(pause_length, dtype=np.float32)
fade_duration = 0.05
fade_length = int(fade_duration * sample_rate)
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
                faded_audio = apply_fade(audio, fade_length)
                audio_segments.append(faded_audio)
            except Exception as e:
                print(f"Error generating audio for chunk {idx+1}: {e}")
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
