# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

from transformers import XLMRobertaConfig
from modeling_roberta import (
    RobertaForQuestionAnsweringAVPool, 
    RobertaForQuestionAnsweringAVPoolBCE, 
    RobertaForQuestionAnsweringSeqSC,
    RobertaForQuestionAnsweringAVPoolBCEV3,
    RobertaForQuestionAnsweringAVDep
)

class XLMRobertaForQuestionAnsweringAVPool(RobertaForQuestionAnsweringAVPool):
    config_class = XLMRobertaConfig

class XLMRobertaForQuestionAnsweringAVPoolBCE(RobertaForQuestionAnsweringAVPoolBCE):
    config_class = XLMRobertaConfig

class XLMRobertaForQuestionAnsweringSeqSC(RobertaForQuestionAnsweringSeqSC):
    config_class = XLMRobertaConfig

class XLMRobertaForQuestionAnsweringAVPoolBCEV3(RobertaForQuestionAnsweringAVPoolBCEV3):
    config_class = XLMRobertaConfig

class XLMRobertaForQuestionAnsweringAVDep(RobertaForQuestionAnsweringAVDep):
    config_class = XLMRobertaConfig