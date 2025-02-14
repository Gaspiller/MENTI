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

from MedchatLLM import MedchatLLM


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
        
        elif 'pulse' in model.lower():
            self.llm = MedchatLLM()

        
    def generate(self, input: str, history: list[str] = []) -> str:

        if 'gpt-3.5' in self.model.lower():
            print("GPT-3.5-turbo Generating.")
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
        
        elif 'pulse' in self.model.lower():
            # print("PULSE Generating.")
            response = self.llm.generate(input)
            return response


class Embeddings:
    def __init__(self, model: str = 'm3e-base') -> None:
        if 'm3e' in model.lower():
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('../m3e-base')
        elif 'simcse' in model.lower():
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