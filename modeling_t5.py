# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss
from transformers import MT5Config, MT5Tokenizer, T5Tokenizer, T5PreTrainedModel

class MT5ForQuestionAnswering(T5PreTrainedModel):
    pass