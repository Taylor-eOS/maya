import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import soundfile as sf
import numpy as np
import pysbd
from ids import CODE_START_TOKEN_ID, CODE_END_TOKEN_ID, CODE_TOKEN_OFFSET, SNAC_MIN_ID, SNAC_MAX_ID, SNAC_TOKENS_PER_FRAME, SOH_ID, EOH_ID, SOA_ID, BOS_ID, TEXT_EOT_ID

TEMPERATURE = 0.1
TOP_P = 1.0
REPETITION_PENALTY = 1.0
TOKEN_PER_CHAR = 28

class AudioGenerator:
    def __init__(self, model_name="maya-research/maya1", snac_name="hubertsiuzdak/snac_24khz"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.snac_model = SNAC.from_pretrained(snac_name).eval()
        if torch.cuda.is_available():
            self.snac_model = self.snac_model.to(self.device)
        self.segmenter = pysbd.Segmenter(language="en", clean=False)

    def build_prompt(self,description,text):
        soh_token=self.tokenizer.decode([SOH_ID])
        eoh_token=self.tokenizer.decode([EOH_ID])
        soa_token=self.tokenizer.decode([SOA_ID])
        sos_token=self.tokenizer.decode([CODE_START_TOKEN_ID])
        eot_token=self.tokenizer.decode([TEXT_EOT_ID])
        bos_token=self.tokenizer.bos_token
        formatted_text=f'<description="{description}"> {text}'
        prompt=soh_token+bos_token+formatted_text+eot_token+eoh_token+soa_token+sos_token
        return prompt

    def extract_snac_codes(self,token_ids):
        try:
            eos_idx=token_ids.index(CODE_END_TOKEN_ID)
        except ValueError:
            eos_idx=len(token_ids)
        snac_codes=[tid for tid in token_ids[:eos_idx] if SNAC_MIN_ID<=tid<=SNAC_MAX_ID]
        return snac_codes

    def unpack_snac_from_7(self,snac_tokens):
        if not snac_tokens:
            return [[],[],[]]
        if snac_tokens and snac_tokens[-1]==CODE_END_TOKEN_ID:
            snac_tokens=snac_tokens[:-1]
        frames=len(snac_tokens)//SNAC_TOKENS_PER_FRAME
        snac_tokens=snac_tokens[:frames*SNAC_TOKENS_PER_FRAME]
        if frames==0:
            return [[],[],[]]
        l1,l2,l3=[],[],[]
        for i in range(frames):
            slots=snac_tokens[i*7:(i+1)*7]
            l1.append((slots[0]-CODE_TOKEN_OFFSET)%4096)
            l2.extend([(slots[1]-CODE_TOKEN_OFFSET)%4096,(slots[4]-CODE_TOKEN_OFFSET)%4096])
            l3.extend([
                (slots[2]-CODE_TOKEN_OFFSET)%4096,
                (slots[3]-CODE_TOKEN_OFFSET)%4096,
                (slots[5]-CODE_TOKEN_OFFSET)%4096,
                (slots[6]-CODE_TOKEN_OFFSET)%4096,
            ])
        return [l1,l2,l3]

    def estimate_token_count_with_tokenizer(self,text):
        return len(self.tokenizer(text).input_ids)

    def generate_audio(self,text,speaker_description,max_new_tokens_hardcap=32768,temperature=0.9,top_p=0.95,repetition_penalty=1.0):
        prompt=self.build_prompt(speaker_description,text)
        inputs=self.tokenizer(prompt,return_tensors="pt")
        if torch.cuda.is_available():
            inputs={k:v.to(self.device) for k,v in inputs.items()}
        input_len=inputs['input_ids'].shape[1]
        model_pos=getattr(self.model.config,"max_position_embeddings",131072)
        safety_margin=512
        available_new_tokens=max(0,model_pos-input_len-safety_margin)
        if available_new_tokens<64:
            raise ValueError(f"Prompt too long for model context window (input_len={input_len}, model_pos={model_pos})")
        caps=[]
        for c in (4096,8192,16384,max_new_tokens_hardcap,available_new_tokens):
            c=min(c,available_new_tokens)
            if c>0:
                caps.append(c)
        caps=sorted(set(caps))
        generated_snac=None
        generated_audio=None
        last_debug=None
        for cap in caps:
            cap=int(cap)
            with torch.inference_mode():
                try:
                    outputs=self.model.generate(**inputs,max_new_tokens=cap,temperature=temperature,top_p=top_p,repetition_penalty=repetition_penalty,do_sample=True,eos_token_id=CODE_END_TOKEN_ID,pad_token_id=self.tokenizer.pad_token_id)
                except Exception as e:
                    last_debug=f"generation error at cap={cap}: {e}"
                    continue
            generated_ids=outputs[0,input_len:].tolist()
            snac_tokens=self.extract_snac_codes(generated_ids)
            if len(snac_tokens)>=SNAC_TOKENS_PER_FRAME:
                generated_snac=snac_tokens
                break
            with torch.inference_mode():
                try:
                    outputs_greedy=self.model.generate(**inputs,max_new_tokens=cap,temperature=0.0,top_p=1.0,do_sample=False,eos_token_id=CODE_END_TOKEN_ID,pad_token_id=self.tokenizer.pad_token_id)
                except Exception as e:
                    last_debug=f"greedy generation error at cap={cap}: {e}"
                    continue
            generated_ids=outputs_greedy[0,input_len:].tolist()
            snac_tokens=self.extract_snac_codes(generated_ids)
            if len(snac_tokens)>=SNAC_TOKENS_PER_FRAME:
                generated_snac=snac_tokens
                break
            last_debug=f"cap={cap} produced snac_len={len(snac_tokens)}"
        if generated_snac is None:
            raise ValueError(f"Failed to generate valid SNAC tokens. Last debug: {last_debug}")
        if len(generated_snac)<7:
            raise ValueError(f"Not enough SNAC tokens generated for audio: {len(generated_snac)}")
        levels=self.unpack_snac_from_7(generated_snac)
        codes_tensor=[torch.tensor(level,dtype=torch.long,device=self.device).unsqueeze(0) for level in levels]
        with torch.inference_mode():
            z_q=self.snac_model.quantizer.from_codes(codes_tensor)
            audio=self.snac_model.decoder(z_q)[0,0].cpu().numpy()
        audio_abs=np.abs(audio)
        threshold=np.max(audio_abs)*0.01 if audio_abs.size>0 else 0
        if len(audio)>2048:
            start_idx=np.where(audio_abs>threshold)[0]
            if len(start_idx)>0:
                start_idx=max(0,start_idx[0]-512)
                audio=audio[start_idx:]
            else:
                audio=audio[2048:]
        end_idx=np.where(audio_abs>threshold)[0]
        if len(end_idx)>0:
            end_idx=min(len(audio)-1,end_idx[-1]+512)
            audio=audio[:end_idx+1]
        return audio,generated_snac

    def save_audio(self,audio,output_file,sample_rate=24000):
        sf.write(output_file,audio,sample_rate)


if __name__ == "__main__":
    generator = AudioGenerator()
    text = "Frodo and Sam walk to Mordor to return the ring."
    speaker_description = "Professional british male audiobook narrator."
    audio = generator.generate_audio(text, speaker_description)
    generator.save_audio(audio, "output.wav")
    print("Audio saved to output.wav")

