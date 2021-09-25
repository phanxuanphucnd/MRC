# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import os
import glob
import torch
import random
import logging
import numpy as np

from evaluator import eval_squad
from tqdm import tqdm, trange
from torch.utils.data import (
    DataLoader, 
    RandomSampler, 
    SequentialSampler, 
    DistributedSampler
)
from transformers import (
    WEIGHTS_NAME, 
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer
)
from transformers import (
    AdamW, 
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features
)
from transformers.data.processors.squad import (
    SquadV1Processor, 
    SquadV2Processor, 
    SquadResult
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits, 
    compute_predictions_log_probs, 
    squad_evaluate
)

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def tolist(tensor):
    return tensor.detach().cpu().tolist()

def train(args, train_dataset, model, tokenizer):
    