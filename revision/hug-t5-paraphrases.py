"""
This script uses the "humarin/chatgpt_paraphraser_on_T5_base" model from Hugging Face.
Model URL: https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base

Purpose: This code is intended for research purposes only and is not used for any commercial activities.

Author: University of Seoul, Mose Park
"""

'''
temp 0.1
'''

import os
import pandas as pd

from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=2,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.1,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)
    
    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

# 실행 예시
# text = "What are the best places to see in New York?"
# paraphrased_texts = paraphrase(text)
# for idx, paraphrased in enumerate(paraphrased_texts, 1):
#     print(f"Paraphrase {idx}: {paraphrased}")

df = pd.read_csv("hug-ori.csv", encoding='utf-8-sig')
df.columns
df.head()

df = df[['idx', 'text', 'paraphrases', 'category', 'source']]

gen1_results = []
gen2_results = []

for txt in tqdm(df['text'], desc="Generating Paraphrases", total=len(df)):
    txts = paraphrase(txt)
    gen1_results.append(txts[0])
    gen2_results.append(txts[1])

df['0.1-gen1'] = gen1_results
df['0.1-gen2'] = gen2_results

df.to_csv("0.1-hug.csv", encoding='utf-8-sig')
