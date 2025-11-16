import pysbd
import numpy as np
from maya import AudioGenerator

speaker_description = "British audiobook narrator, male, 40s, without accent, reading a nonfiction book professionally, pronouncing clearly, holding pauses only at natural logical breaks."

def process_text_file(input_file, speaker_description):
    generator = AudioGenerator()
    segmenter = pysbd.Segmenter(language="en", clean=False)
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    chapters = text.split('\n\n')
    chapters = [ch.strip() for ch in chapters if ch.strip()]
    print(f"Processing {len(chapters)} chapters")
    for chapter_idx, chapter_text in enumerate(chapters, 1):
        print(f"\n{'='*60}")
        print(f"CHAPTER {chapter_idx}/{len(chapters)}")
        print(f"{'='*60}")
        sentences = segmenter.segment(chapter_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        print(f"Processing {len(sentences)} sentences")
        audio_segments = []
        for i, sentence in enumerate(sentences, 1):
            print(f"\nSentence {i}/{len(sentences)}: {sentence[:50]}...")
            audio = generator.generate_audio(sentence, speaker_description)
            audio_segments.append(audio)
        combined_audio = np.concatenate(audio_segments)
        output_file = f"output_chapter_{chapter_idx:03d}.wav"
        generator.save_audio(combined_audio, output_file)
        print(f"\nSaved chapter {chapter_idx} to {output_file}")
        print(f"Duration: {len(combined_audio)/24000:.2f} seconds")

if __name__ == "__main__":
    process_text_file("input.txt", speaker_description)

