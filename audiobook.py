import torch
import numpy as np
import os
from pysbd import Segmenter
from maya import AudioGenerator

gen = AudioGenerator()
segmenter = Segmenter(language="en", clean=False)
speaker_description = "Realistic male audiobook narrator in the 50s age with british accent. Normal pitch, warm timbre, friendly tone, reading pacing, good pronounciation."
fixed_seed = 42
max_input_tokens = 1024
max_new_tokens = 16384
sample_rate = 24000
output_dir = "."
os.makedirs(output_dir, exist_ok=True)

desc_prompt = gen.build_prompt(speaker_description, "")
desc_inputs = gen.tokenizer(desc_prompt, return_tensors="pt")
desc_tokens = desc_inputs["input_ids"].shape[1]

def split_and_generate(text, speaker_description, max_tokens, fixed_seed, gen, segmenter, desc_tokens):
    if not text.strip():
        return []
    prompt = gen.build_prompt(speaker_description, text)
    inputs = gen.tokenizer(prompt, return_tensors="pt")
    token_count = inputs["input_ids"].shape[1]
    if token_count <= max_tokens:
        torch.manual_seed(fixed_seed)
        try:
            audio = gen.generate_audio(text, speaker_description, max_new_tokens)
            return [audio]
        except ValueError as e:
            print(f"Error generating audio for text: {e}")
            return []
    sentences = segmenter.segment(text)
    if len(sentences) <= 1:
        words = text.split()
        subchunks = []
        current_subchunk = []
        for word in words:
            temp_subchunk = current_subchunk + [word]
            temp_text = " ".join(temp_subchunk)
            sub_prompt = gen.build_prompt(speaker_description, temp_text)
            sub_inputs = gen.tokenizer(sub_prompt, return_tensors="pt")
            sub_token_count = sub_inputs["input_ids"].shape[1]
            if sub_token_count > max_tokens:
                if current_subchunk:
                    subchunks.append(" ".join(current_subchunk))
                current_subchunk = [word]
            else:
                current_subchunk = temp_subchunk
        if current_subchunk:
            subchunks.append(" ".join(current_subchunk))
        audios = []
        for subchunk_text in subchunks:
            torch.manual_seed(fixed_seed)
            try:
                sub_audio = gen.generate_audio(subchunk_text, speaker_description, max_new_tokens)
                audios.append(sub_audio)
            except ValueError as e:
                print(f"Error generating subchunk: {e}")
        return audios
    var_tokens_per_sent = []
    for sentence in sentences:
        sent_prompt = gen.build_prompt(speaker_description, sentence)
        sent_inputs = gen.tokenizer(sent_prompt, return_tensors="pt")
        full_sent_tokens = sent_inputs["input_ids"].shape[1]
        var_token = full_sent_tokens - desc_tokens
        var_tokens_per_sent.append(var_token)
    cum_var = np.cumsum(var_tokens_per_sent)
    total_var = cum_var[-1]
    half_var = total_var / 2.0
    split_idx = np.argmin(np.abs(cum_var - half_var))
    text1 = " ".join(sentences[:split_idx + 1])
    text2 = " ".join(sentences[split_idx + 1 :])
    audios1 = split_and_generate(text1, speaker_description, max_tokens, fixed_seed, gen, segmenter, desc_tokens)
    audios2 = split_and_generate(text2, speaker_description, max_tokens, fixed_seed, gen, segmenter, desc_tokens)
    return audios1 + audios2

with open("input.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

sections = [section.strip() for section in full_text.split("\n\n") if section.strip()]

for section_index, section in enumerate(sections):
    full_text_for_check = " ".join(section.splitlines())
    audio_segments = split_and_generate(full_text_for_check, speaker_description, max_input_tokens, fixed_seed, gen, segmenter, desc_tokens)
    if audio_segments:
        full_audio = np.concatenate(audio_segments)
        output_file = os.path.join(output_dir, f"output_{section_index + 1}.wav")
        gen.save_audio(full_audio, output_file, sample_rate)
        print(f"Saved audio for section {section_index + 1} to {output_file}")
    else:
        print(f"No audio generated for section {section_index + 1}")
