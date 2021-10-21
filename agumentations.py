

def translate_marian(texts, model, tokenizer, language="en"):
    if isinstance(texts, str):
        texts = [texts]
    # Prepare the text data into appropriate format for the model
    template = lambda text: f"{text}" if language == "vi" else f">>{language}<< {text}"
    src_texts = [template(text) for text in texts]
    
    print("\nTranslate: ", src_texts)

    # Tokenize the texts
    encoded = tokenizer.prepare_seq2seq_batch(src_texts, return_tensors='pt')
    
    # Generate translation using model
    translated = model.generate(**encoded)

    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    
    return translated_texts

def back_translate_marian(texts, tar_model, tar_tokenizer, enc_model, enc_tokenizer, source_lang="vi", target_lang="en"):
    # Translate from source to target language
    fr_texts = translate_marian(texts, tar_model, tar_tokenizer, 
                         language=target_lang)

    # Translate from target language back to source language
    outputs = translate_marian(fr_texts, enc_model, enc_tokenizer, 
                                      language=source_lang)
    
    return outputs

def translate_t5(src_text, tar_model, tar_tokenizer):
    # src = "Vị trí địa lý của Pháp có gì đặc biệt?"
    tokenized_text = tar_tokenizer.encode(src_text, return_tensors="pt").to(device)
    tar_model.eval()
    summary_ids = tar_model.generate(
                        tokenized_text,
                        max_length=128, 
                        num_beams=5,
                        repetition_penalty=2.5, 
                        length_penalty=1.0, 
                        early_stopping=True
                    )
    output = tar_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return output

def back_translate_t5(src_text, tar_model, tar_tokenizer, enc_model, enc_tokenizer):

    tar_text = translate_t5(src_text, tar_model, tar_tokenizer)

    tokenized_text = enc_tokenizer.encode(tar_text, return_tensors="pt").to(device)
    enc_model.eval()
    summary_ids = enc_model.generate(
                        tokenized_text,
                        max_length=128, 
                        num_beams=5,
                        repetition_penalty=2.5, 
                        length_penalty=1.0, 
                        early_stopping=True
                    )
    output = enc_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return output

def translate(text, tar_model, tar_tokenizer, enc_model, enc_tokenizer):    
    
    output = back_translate_t5(text, tar_model, tar_tokenizer, enc_model, enc_tokenizer)

    return output

import enum
import json
import random

from agumentations import *
from tqdm import tqdm, trange

import torch

from transformers import MarianMTModel, MarianTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

name_file = 'train-0'
with open(f'data/uit-visquad/{name_file}.json', 'r+', encoding='utf-8') as f:
    data = json.load(f)

data_ = data['data']

tar_model = T5ForConditionalGeneration.from_pretrained("NlpHUST/t5-vi-en-base")
tar_tokenizer = T5Tokenizer.from_pretrained("NlpHUST/t5-vi-en-base")
tar_model.to(device)

enc_model = T5ForConditionalGeneration.from_pretrained("NlpHUST/t5-en-vi-base")
enc_tokenizer = T5Tokenizer.from_pretrained("NlpHUST/t5-en-vi-base")
enc_model.to(device)

iterator = trange(int(len(data_)), desc="Paragraph: ")

for i, _ in enumerate(iterator):
    paragraphs = data_[i]['paragraphs']
    for par in paragraphs:
        qas = par['qas']
        len_qas = len(qas)

        for j in tqdm(range(len_qas)):
            q = qas[j]
            if q['is_impossible']:
                tmp = {
                    'id': q['id'] + str(j), 
                    'answers': q['answers'],
                    'is_impossible': q['is_impossible']
                }
                question = q['question']
                aug_text = translate(question, tar_model, tar_tokenizer, enc_model, enc_tokenizer)
                tmp['question'] = aug_text
                
                if aug_text.lower() != question.lower():
                    qas.append(aug_text)

with open(f'data/uit-visquad/{name_file}_aug.json', 'w', encoding='utf-8') as f:
    json.dump(data, f)