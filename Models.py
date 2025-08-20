import json
import tqdm
import torch
import argparse
import transformers
import os

from transformers import AutoTokenizer, AutoModel
from retry import retry
from tqdm import tqdm
from openai import OpenAI

# from MedchatLLM import MedchatLLM


class LLMs:
    def __init__(self, model: str = 'GPT-3.5-Turbo', device: torch.device = torch.device('cuda:0')) -> None:

        self.model = model

        if 'gpt-3.5' in model.lower():
            # GPT-3.5-Turbo
            self.client = OpenAI()

        if 'gpt-4' in model.lower():
            # GPT-4o or GPT-4
            self.client = OpenAI()

        elif 'chatglm3' in model.lower():
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, device_map='auto')
            model = AutoModel.from_pretrained(model, trust_remote_code=True, device_map='auto').half()
            model = model.eval()
            self.llm = model
            self.tokenizer = tokenizer

        elif 'bianque' in model.lower():
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, device_map='auto')
            model = AutoModel.from_pretrained(model, trust_remote_code=True, device_map='auto').half()
            model = model.eval()
            self.llm = model
            self.tokenizer = tokenizer
        
        # elif 'pulse' in model.lower():
        #     self.llm = MedchatLLM()

        
    def generate(self, input: str, history: list[str] = []) -> str:

        if 'gpt-3.5' in self.model.lower():
            # print("GPT-3.5-turbo Generating.")
            # @retry(tries=-1)
            def _chat(messages):
                response = self.client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
                return response
            messages = []
            messages.append({"role": "user", "content": input})
            ans = _chat(messages).choices[0].message.content
            return ans

        elif 'gpt-4o' == self.model.lower():
            # print("GPT-4o Generating.")
            @retry(tries=-1)
            def _chat(messages):
                response = self.client.chat.completions.create(model="gpt-4o", messages=messages)
                return response
            messages = []
            messages.append({"role": "user", "content": input})
            
            ans = _chat(messages).choices[0].message.content
            return ans
        
        elif 'gpt-4' == self.model.lower():
            print("GPT-4 Generating.")
            @retry(tries=-1)
            def _chat(messages):
                # print("   testing...")
                response = self.client.chat.completions.create(model="gpt-4", messages=messages)
                return response
            messages = []
            messages.append({"role": "user", "content": input})
            ans = _chat(messages).choices[0].message.content
            return ans
            
        elif 'chatglm3' in self.model.lower():
            response, history = self.llm.chat(self.tokenizer, input, history=history)
            return response
        
        elif 'bianque' in self.model.lower():
            response, history = self.llm.chat(self.tokenizer, query=input, history=history, max_length=2048, num_beams=1, do_sample=True, top_p=0.75, temperature=0.95, logits_processor=None)
            return response
        
        # elif 'pulse' in self.model.lower():
        #     # print("PULSE Generating.")
        #     response = self.llm.generate(input)
        #     return response


class Embeddings:
    def __init__(self, model: str = 'm3e-base') -> None:
        if 'm3e' in (model or '').lower():
            from sentence_transformers import SentenceTransformer
            base_dir = os.path.dirname(__file__)
            user_spec = (model or '').strip()

            candidates = []

            # 1) 用户传入的是本地目录（绝对/相对），直接使用
            if user_spec and os.path.exists(user_spec):
                candidates.append(user_spec)

            # 2) 用户传入名称时，按本地常见目录匹配
            def add_name(name: str):
                candidates.extend([
                    os.path.join(base_dir, name),
                    os.path.join(os.getcwd(), 'MENTI', name),
                    '/app/' + name,
                ])

            lower = user_spec.lower()
            if lower == 'm3e-large':
                add_name('m3e-large')
            elif lower == 'm3e-base':
                add_name('m3e-base')
            else:
                # 传 'm3e' 或其它：按 large -> base 顺序尝试
                add_name('m3e-large')
                add_name('m3e-base')

            # 去重并加载
            seen = set()
            uniq = []
            for c in candidates:
                if c and (c not in seen):
                    uniq.append(c)
                    seen.add(c)

            emb = None
            last_err = None
            for cand in uniq:
                if not os.path.exists(cand):
                    continue
                try:
                    emb = SentenceTransformer(cand)
                    break
                except Exception as e:
                    last_err = e
                    continue

            if emb is None:
                tried = [c for c in uniq if c]
                raise FileNotFoundError(f"未找到可用的本地 m3e 模型目录，已尝试：{tried}")

            self.embedding_model = emb
        elif 'simcse' in (model or '').lower():
            from simcse import SimCSE
            self.embedding_model = SimCSE("../sup-simcse-bert-base-uncased")


def set_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = set_configs()
    llm = LLMs(args.model)

    print(llm.generate("Hello."))