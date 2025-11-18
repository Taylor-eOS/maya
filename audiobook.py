import pysbd
import numpy as np
import time
from maya import AudioGenerator

speaker_description = "A British male without accent who is a professional audiobook narrator in his late forties with a deep, resonant voice, speaking clearly in a calm, measured, and engaged tone."

def process_text_file(input_file, speaker_description):
    generator = AudioGenerator()
    segmenter = pysbd.Segmenter(language="en", clean=False)
    debug_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    chapters = text.split('\n\n')
    chapters = [ch.strip() for ch in chapters if ch.strip()]
    print(f"Processing {len(chapters)} chapters")
    for chapter_idx, chapter_text in enumerate(chapters, 1):
        chapter_start = time.time()
        print(f"CHAPTER {chapter_idx}/{len(chapters)}")
        sentences = segmenter.segment(chapter_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        print(f"Processing {len(sentences)} sentences")
        audio_segments = []
        sentence_idx = 1
        for sentence in sentences:
            if not sentence:
                continue
            sentence_start = time.time()
            print(f"Sentence {sentence_idx}/{len(sentences)}: \"{sentence}\"")
            audio, snac_tokens = generator.generate_audio(sentence, speaker_description)
            audio_segments.append(audio)
            char_count = len(sentence)
            snac_count = len(snac_tokens)
            ratio = snac_count / char_count if char_count > 0 else 0
            print(f"Characters: {char_count}, SNAC tokens: {snac_count}, Ratio: {ratio:.2f}")
            debug_data.append(f"Chapter {chapter_idx}, Sentence {sentence_idx}: chars={char_count}, snac={snac_count}, ratio={ratio:.2f}")
            sentence_idx += 1
            sentence_time = time.time() - sentence_start
            print(f"Time: {sentence_time:.0f}s, {sentence_time/char_count:.1f}char/s")
        combined_audio = np.concatenate(audio_segments)
        output_file = f"output_chapter_{chapter_idx:03d}.wav"
        generator.save_audio(combined_audio, output_file)
        chapter_time = time.time() - chapter_start
        print(f"Saved chapter {chapter_idx} to {output_file}")
        print(f"Duration: {len(combined_audio)/24000:.2f} seconds")
        print(f"Chapter processing time: {chapter_time:.2f}s")
    if debug_data:
        for line in debug_data:
            print(line)
        total_ratios = [float(line.split('ratio=')[1]) for line in debug_data]
        avg_ratio = sum(total_ratios) / len(total_ratios)
        print(f"Overall average tokens per character: {avg_ratio:.2f}")

if __name__ == "__main__":
    print(f"Speaker: {speaker_description}")
    process_text_file("input.txt", speaker_description)

