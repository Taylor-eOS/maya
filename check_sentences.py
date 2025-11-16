import os
from pysbd import Segmenter

def find_longest_sentences(file_path="input.txt", top_n=2):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        print("File is empty.")
        return
    segmenter = Segmenter(language="en", clean=False)
    sentences = segmenter.segment(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        print("No sentences found.")
        return
    sentences_with_lengths = [(s, len(s.split())) for s in sentences]
    sentences_with_lengths.sort(key=lambda x: x[1], reverse=True)
    print(f"Found {len(sentences)} sentences in total.")
    print(f"Showing the {top_n} longest by word count:\n")
    for i, (sentence, word_count) in enumerate(sentences_with_lengths[:top_n], 1):
        print(f"{i}. {word_count} words:")
        print(f"   {sentence}\n")

if __name__ == "__main__":
    find_longest_sentences()
