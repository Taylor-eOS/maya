import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import soundfile as sf
import numpy as np

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

def build_prompt(tokenizer, description: str, text: str) -> str:
    soh_token = tokenizer.decode([SOH_ID])
    eoh_token = tokenizer.decode([EOH_ID])
    soa_token = tokenizer.decode([SOA_ID])
    sos_token = tokenizer.decode([CODE_START_TOKEN_ID])
    eot_token = tokenizer.decode([TEXT_EOT_ID])
    bos_token = tokenizer.bos_token
    formatted_text = f'<description="{description}"> {text}'
    prompt = (
        soh_token + bos_token + formatted_text + eot_token +
        eoh_token + soa_token + sos_token
    )
    return prompt

def extract_snac_codes(token_ids: list) -> list:
    try:
        eos_idx = token_ids.index(CODE_END_TOKEN_ID)
    except ValueError:
        eos_idx = len(token_ids)
    snac_codes = [
        token_id for token_id in token_ids[:eos_idx]
        if SNAC_MIN_ID <= token_id <= SNAC_MAX_ID
    ]
    return snac_codes

def unpack_snac_from_7(snac_tokens: list) -> list:
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
        l2.extend([
            (slots[1] - CODE_TOKEN_OFFSET) % 4096,
            (slots[4] - CODE_TOKEN_OFFSET) % 4096,
        ])
        l3.extend([
            (slots[2] - CODE_TOKEN_OFFSET) % 4096,
            (slots[3] - CODE_TOKEN_OFFSET) % 4096,
            (slots[5] - CODE_TOKEN_OFFSET) % 4096,
            (slots[6] - CODE_TOKEN_OFFSET) % 4096,
        ])
    return [l1, l2, l3]

def main():
    print("\n[1/3] Loading Maya1 model")
    model = AutoModelForCausalLM.from_pretrained("maya-research/maya1", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("maya-research/maya1", trust_remote_code=True)
    print(f"Model loaded: {len(tokenizer)} tokens in vocabulary")
    print("\n[2/3] Loading SNAC audio decoder...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    if torch.cuda.is_available():
        snac_model = snac_model.to("cuda")
    print("SNAC decoder loaded")
    description = "British male audio book narration, serious and professional."
    text = "Frodo and Sam walk to Mordor to return the ring."
    print("\n[3/3] Generating speech...")
    print(f"Description: {description}")
    print(f"Text: {text}")
    prompt = build_prompt(tokenizer, description, text)
    print(f"\nPrompt preview (first 200 chars):")
    print(f"   {repr(prompt[:200])}")
    print(f"   Prompt length: {len(prompt)} chars")
    inputs = tokenizer(prompt, return_tensors="pt")
    print(f"   Input token count: {inputs['input_ids'].shape[1]} tokens")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=2048,  
            min_new_tokens=28,  
            temperature=0.4, 
            top_p=0.9, 
            repetition_penalty=1.1,  
            do_sample=True,
            eos_token_id=CODE_END_TOKEN_ID,  
            pad_token_id=tokenizer.pad_token_id,
        )
    generated_ids = outputs[0, inputs['input_ids'].shape[1]:].tolist()
    print(f"Generated {len(generated_ids)} tokens")
    print(f"   First 20 tokens: {generated_ids[:20]}")
    print(f"   Last 20 tokens: {generated_ids[-20:]}")
    if CODE_END_TOKEN_ID in generated_ids:
        eos_position = generated_ids.index(CODE_END_TOKEN_ID)
        print(f" EOS token found at position {eos_position}/{len(generated_ids)}")
    snac_tokens = extract_snac_codes(generated_ids)
    print(f"Extracted {len(snac_tokens)} SNAC tokens")
    snac_count = sum(1 for t in generated_ids if SNAC_MIN_ID <= t <= SNAC_MAX_ID)
    other_count = sum(1 for t in generated_ids if t < SNAC_MIN_ID or t > SNAC_MAX_ID)
    print(f"   SNAC tokens in output: {snac_count}")
    print(f"   Other tokens in output: {other_count}")
    if CODE_START_TOKEN_ID in generated_ids:
        sos_pos = generated_ids.index(CODE_START_TOKEN_ID)
        print(f"   SOS token at position: {sos_pos}")
    else:
        print(f"   No SOS token found in generated output!")
    if len(snac_tokens) < 7:
        print("Error: Not enough SNAC tokens generated")
        return
    levels = unpack_snac_from_7(snac_tokens)
    frames = len(levels[0])
    print(f"Unpacked to {frames} frames")
    print(f"   L1: {len(levels[0])} codes")
    print(f"   L2: {len(levels[1])} codes")
    print(f"   L3: {len(levels[2])} codes")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    codes_tensor = [
        torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0)
        for level in levels
    ]
    print("\n[4/4] Decoding to audio...")
    with torch.inference_mode():
        z_q = snac_model.quantizer.from_codes(codes_tensor)
        audio = snac_model.decoder(z_q)[0, 0].cpu().numpy()
    if len(audio) > 2048:
        audio = audio[2048:]
    duration_sec = len(audio) / 24000
    print(f"Audio generated: {len(audio)} samples ({duration_sec:.2f}s)")
    output_file = "output.wav"
    sf.write(output_file, audio, 24000)
    print(f"\nVoice generated successfully!")

if __name__ == "__main__":
    main()

