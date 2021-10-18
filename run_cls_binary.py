import pandas as pd
from models import *
from tqdm import tqdm
tqdm.pandas()
from torch import nn
import json
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import *
import torch
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F
import argparse
from transformers.modeling_utils import * 
from utils import *
from modeling_xlm_roberta import XLMRobertaForBinaryClassification, RobertaForBinaryClassification

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_path', type=str, default='./data/train_cmt.csv')
parser.add_argument('--model_name_or_path', type=str)
parser.add_argument('--max_sequence_length', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--accumulation_steps', type=int, default=5)
parser.add_argument("--cache_dir", default="", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--do_lower_case", type=bool, default=True,
                    help="Set this flag if you are using an uncased model.")
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--seed', type=int, default=69)
parser.add_argument('--ckpt_path', type=str, default='./models/cls')
parser.add_argument('--lr', type=float, default=3e-5)

args = parser.parse_args()

seed_everything(69)

# Load model
config = RobertaConfig.from_pretrained(
    args.model_name_or_path, output_hidden_states=True, 
    num_labels=1, cache_dir=args.cache_dir if args.cache_dir else None)

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

model = RobertaForBinaryClassification.from_pretrained(
    args.model_name_or_path, config=config,
    from_tf=bool('.ckpt' in args.model_name_or_path),
    cache_dir=args.cache_dir if args.cache_dir else None)

model.cuda()

if torch.cuda.device_count():
    print(f"Training using {torch.cuda.device_count()} gpus")
    model = nn.DataParallel(model)
    tsfm = model.module.roberta if hasattr(model, 'module') else model.roberta
else:
    tsfm = model.roberta


# Load training data
train_df = pd.read_csv(args.train_path, encoding='utf-8')

y = train_df.label.values
X_train = convert_lines(train_df, tokenizer, args.max_sequence_length)

# Creating optimizer and lr schedulers
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
num_train_optimization_steps = int(args.epochs*len(train_df)/args.batch_size/args.accumulation_steps)
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
scheduler0 = get_constant_schedule(optimizer)  # PyTorch scheduler

if not os.path.exists(args.ckpt_path):
    os.mkdir(args.ckpt_path)

splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(X_train, y))
for fold, (train_idx, val_idx) in enumerate(splits):
    print("Training for fold {}".format(fold))
    best_score = 0
    if fold != args.fold:
        continue
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train[train_idx],dtype=torch.long), torch.tensor(y[train_idx],dtype=torch.long))
    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train[val_idx],dtype=torch.long), torch.tensor(y[val_idx],dtype=torch.long))
    tq = tqdm(range(args.epochs + 1))
    for child in tsfm.children():
        for param in child.parameters():
            if not param.requires_grad:
                print("whoopsies")
            param.requires_grad = False
    frozen = True
    for epoch in tq:

        if epoch > 0 and frozen:
            for child in tsfm.children():
                for param in child.parameters():
                    param.requires_grad = True
            frozen = False
            del scheduler0
            torch.cuda.empty_cache()

        val_preds = None
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        avg_loss = 0.
        avg_accuracy = 0.

        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader),total=len(train_loader),leave=False)
        for i,(x_batch, y_batch) in pbar:
            model.train()
            y_pred = model(x_batch.cuda(), attention_mask=(x_batch>0).cuda())
            loss =  F.binary_cross_entropy_with_logits(y_pred.view(-1).cuda(), y_batch.float().cuda())
            loss = loss.mean()
            loss.backward()
            if i % args.accumulation_steps == 0 or i == len(pbar) - 1:
                optimizer.step()
                optimizer.zero_grad()
                if not frozen:
                    scheduler.step()
                else:
                    scheduler0.step()
            lossf = loss.item()
            pbar.set_postfix(loss = lossf)
            avg_loss += loss.item() / len(train_loader)

        model.eval()
        pbar = tqdm(enumerate(valid_loader),total=len(valid_loader),leave=False)
        for i,(x_batch, y_batch) in pbar:
            y_pred = model(x_batch.cuda(), attention_mask=(x_batch>0).cuda())
            y_pred = y_pred.squeeze().detach().cpu().numpy()
            val_preds = np.atleast_1d(y_pred) if val_preds is None else np.concatenate([val_preds, np.atleast_1d(y_pred)])
        val_preds = sigmoid(val_preds)

        best_th = 0
        score = f1_score(y[val_idx], val_preds > 0.5)
        print(f"\nAUC = {roc_auc_score(y[val_idx], val_preds):.4f}, F1 score @0.5 = {score:.4f}")
        if score >= best_score:
            torch.save(model.state_dict(),os.path.join(args.ckpt_path, f"model_{fold}.bin"))
            best_score = score