import os
import re
import glob
import json
import time
import torch
import random
import logging
import argparse
import numpy as np
import pandas as pd

import pytorch_lightning as pl

from pathlib import Path
from itertools import chain
from string import punctuation
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import (
    AdamW,
    MT5ForConditionalGeneration,
    MT5Tokenizer, 
    get_linear_schedule_with_warmup
)

pl.seed_everything(0)

def extract_questions_and_answers(data_path):
    with open(data_path, 'r+', encoding='utf-8') as json_file:
        data = json.load(json_file)
        data_ = data['data']
        data_rows = []

        for i in range(len(data_)):
            paragraphs = data_[i]['paragraphs']
            for par in paragraphs:
                context = par['context']
                qas = par['qas']
                for qa in qas:
                    question = qa['question']
                    answers = qa.get('answers', "")

                    if not qa['is_impossible']:
                        for answer in answers:
                            answer_text = answer['text']
                            answer_start = answer['answer_start']
                            answer_end = answer['answer_start'] + len(answer_text)  #Gets the end index of each answer in the paragraph
                            data_rows.append({
                                "question" : question,
                                "context"  : context,
                                "answer_text" : answer_text,
                                "answer_start" : answer_start,
                                "answer_end" : answer_end,
                                "no_answer": 0
                            })
                    else:
                        # print(answer_text, answer_start)
                        data_rows.append({
                            "question" : question,
                            "context"  : context,
                            "answer_text" : "",
                            "answer_start" : None,
                            "answer_end" : None,
                            "no_answer": 1
                        })

    return pd.DataFrame(data_rows)

train_path = 'data/uit-visquad/train-0.json'

train_df = extract_questions_and_answers(train_path)

dev_path = 'data/uit-visquad/dev-0.json'

dev_df = extract_questions_and_answers(dev_path)

MODEL_NAME ='google/mt5-base'

tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)

class UITMRCDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: MT5Tokenizer,
        source_max_token_len: int=396,
        target_max_token_len: int=50,
    ):
        self.data =  data
        self.tokenizer =  tokenizer
        self.source_max_token_len =  source_max_token_len
        self.target_max_token_len =  target_max_token_len


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        source_encoding = tokenizer(
            data_row['question'],
            data_row['context'],
            max_length=self.source_max_token_len,
            padding='max_length',
            truncation="only_second",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        target_encoding = tokenizer(
            data_row['answer_text'],
            max_length=self.target_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        labels = target_encoding['input_ids']
        labels[labels==0] = -100

        return dict(
            question=data_row['question'],
            context=data_row['context'],
            answer_text=data_row['answer_text'],
            input_ids=source_encoding["input_ids"].flatten(),
            attention_mask=source_encoding['attention_mask'].flatten(),
            labels=labels.flatten()
        )


class UITMRCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer:MT5Tokenizer,
        batch_size: int = 8,
        source_max_token_len: int=396,
        target_max_token_len: int=50,
    ):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def setup(self):
        self.train_dataset = UITMRCDataset(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

        self.test_dataset = UITMRCDataset(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=4
        )
    
BATCH_SIZE = 8
N_EPOCHS = 10

data_module = UITMRCDataModule(train_df, dev_df, tokenizer, batch_size=BATCH_SIZE)
data_module.setup()

model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict = True)

print(model.config)

class MT5ForQuestionAnswering(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)


    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids, 
            attention_mask=attention_mask,
            labels=labels)

        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask=batch['attention_mask']
        labels = batch['labels']
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask=batch['attention_mask']
        labels = batch['labels']
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask=batch['attention_mask']
        labels = batch['labels']
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=0.0001)
        
        return optimizer

model = MT5ForQuestionAnswering()

# To record the best performing model using checkpoint

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=2,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

trainer = pl.Trainer(
    #logger = logger,
    callbacks=checkpoint_callback,
    max_epochs=N_EPOCHS,
    gpus=1,
    progress_bar_refresh_rate = 30
)

trainer.fit(model, data_module)

trainer.test()  # evaluate the model according to the last checkpoint