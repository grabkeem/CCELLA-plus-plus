# This script is used to generate the modified JSON files and text embeddings

import os
import pandas as pd
import io
import json
from tqdm import tqdm
import torch
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np

path = "[path to data]"

processed_datafolder = path+"/coreg_diffusion_new"
embedding_base_dir = processed_datafolder+"_emb"
raw_datafolder = path+"/pre_coreg_diffusion"

# Output filenames
train_files = './maisi/datasets/train_merged.json'
test_files = './maisi/datasets/test_merged.json'

train_in = './maisi/datasets/train.json'
test_in = './maisi/datasets/test.json'


train_dict = json.load(train_in)
test_dict = json.load(test_in)

sample_flag = True

pretrained_path="google/flan-t5-xxl"
device='cuda:0'
model = T5EncoderModel.from_pretrained(pretrained_path).to(device)
tokenizer = T5Tokenizer.from_pretrained(pretrained_path, truncation_side='left')

with torch.no_grad():
    null_text = ''
    inputs = tokenizer.encode_plus(null_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    inputs_input_ids = inputs['input_ids'].to(device)
    inputs_attention_mask = inputs['attention_mask'].to(device)
    output_np = model(inputs_input_ids, attention_mask=inputs_attention_mask).last_hidden_state.cpu().numpy().squeeze(0)
    np.save(f'{embedding_base_dir}/null.npy', output_np)
    del output_np
    del inputs_input_ids
    del inputs_attention_mask
    del inputs


    for entry in tqdm(train_dict):
        # Find entry['folder'] in df
        text_isnull = False        

        # Add the text to the entry
        if train_dict['text']=='':
            entry_text = ''
            text_isnull = True
        else:
            entry_text = train_dict['text']

        entry_text = tokenizer.encode_plus(entry_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        entry_text_input_ids = entry_text['input_ids'].to(device)
        entry_text_attention_mask = entry_text['attention_mask'].to(device)
        output = model(entry_text_input_ids, attention_mask=entry_text_attention_mask).last_hidden_state.cpu().numpy().squeeze(0)
        output_textfile = f'{embedding_base_dir}/{entry["folder"]}/text.npy'
        np.save(output_textfile, output)
        
        if sample_flag:
            print(entry['folder'])
            print(entry_text)
            sample_flag = False
            print(output.shape)

        entry['text'] = output_textfile
        entry['text_isnull'] = int(text_isnull)

        del entry_text
        del entry_text_input_ids
        del entry_text_attention_mask
        del output

    with open(train_files, 'w') as f:
        json.dump(train_dict, f)

    sample_flag = True

    for entry in tqdm(test_dict):
        # Find entry['folder'] in df
        text_isnull = False
        
        # Add the text to the entry
        if train_dict['text']=='':
            entry_text = ''
            text_isnull = True
        else:
            entry_text = train_dict['text']

        entry_text = tokenizer.encode_plus(entry_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        entry_text_input_ids = entry_text['input_ids'].to(device)
        entry_text_attention_mask = entry_text['attention_mask'].to(device)
        output = model(entry_text_input_ids, attention_mask=entry_text_attention_mask).last_hidden_state.cpu().numpy().squeeze(0)
        output_textfile = f'{embedding_base_dir}/{entry["folder"]}/text.npy'
        np.save(output_textfile, output)

        if sample_flag:
            print(entry['folder'])
            print(entry_text)
            sample_flag = False
            print(output.shape)
        
        entry['text'] = output_textfile
        entry['text_isnull'] = int(text_isnull)

        del entry_text
        del entry_text_input_ids
        del entry_text_attention_mask
        del output

    with open(test_files, 'w') as f:
        json.dump(test_dict, f)

print("Done")