
# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import math
import torch
import torch.nn as nn

from transformers import *
from torch.nn import CrossEntropyLoss, MSELoss
from modules import SCAttention, split_ques_context, TrmCoAtt
from transformers.models.electra.modeling_electra import ElectraPreTrainedModel, ElectraModel

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class ElectraForQuestionAnsweringAVPool(ElectraPreTrainedModel):
    def __init__(self, config, no_answer_loss_coef:float=1.0, ):
        super(ElectraForQuestionAnsweringAVPool, self).__init__(config)
        self.num_labels = config.num_labels
        self.no_answer_loss_coef = no_answer_loss_coef

        self.electra = ElectraModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.has_ans = nn.Sequential(nn.Dropout(p=config.hidden_dropout_prob), nn.Linear(config.hidden_size, 2))
        
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None, output_attentions=None,
            output_hidden_states=None,):

        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        first_word = sequence_output[:, 0, :]

        has_log = self.has_ans(first_word)

        outputs = (start_logits, end_logits, has_log,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(is_impossibles.size()) > 1:
                is_impossibles = is_impossibles.squeeze(-1)
            
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            is_impossibles.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            choice_loss = loss_fct(has_log, is_impossibles)
            total_loss = (start_loss + end_loss + self.no_answer_loss_coef * choice_loss) / 3
            outputs = (total_loss,) + outputs
            # print(sum(is_impossibles==1),sum(is_impossibles==0))cd 
        
        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class ElectraForQuestionAnsweringAVDep(ElectraPreTrainedModel):
    def __init__(
        self, 
        config, 
        start_coef: float=0.3, 
        end_coef: float=0.3,
        has_ans_coef: float=0.4, 
    ):
        super(ElectraForQuestionAnsweringAVDep, self).__init__(config)
        
        self.num_labels = config.num_labels
        self.start_coef = start_coef
        self.end_coef = end_coef
        self.has_ans_coef = has_ans_coef
        
        self.electra = ElectraModel(config)
        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_pooler = nn.Linear(1 + config.hidden_size, 512)
        self.end_outputs = nn.Linear(512, 1)
        self.has_ans = nn.Sequential(nn.Dropout(p=config.hidden_dropout_prob), nn.Linear(config.hidden_size, 2))
        
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None, output_attentions=None,
                output_hidden_states=None,):

        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        start_logits = self.start_outputs(sequence_output)
        start_logits = start_logits.squeeze(-1)
        # batch, seq
        start_logits = start_logits.unsqueeze(-1)

        final_repr = gelu(self.end_pooler(torch.cat([start_logits, sequence_output], dim=-1)))
        end_logits = self.end_outputs(final_repr)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        first_word = sequence_output[:, 0, :]
        has_log = self.has_ans(first_word)

        outputs = (start_logits, end_logits, has_log,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(is_impossibles.size()) > 1:
                is_impossibles = is_impossibles.squeeze(-1)
            
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            is_impossibles.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            choice_loss = loss_fct(has_log, is_impossibles)
            
            if self.start_coef and self.end_coef and self.has_ans_coef:
                total_loss = self.start_coef*start_loss + self.end_coef*end_loss + self.has_ans_coef*choice_loss
            else:
                total_loss = (start_loss + end_loss + self.has_ans_coef * choice_loss) / 3
            
            outputs = (total_loss,) + outputs
        
        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class ElectraForQuestionAnsweringAVDep2(ElectraPreTrainedModel):
    def __init__(
        self, 
        config, 
        start_coef: float=0.3, 
        end_coef: float=0.3,
        has_ans_coef: float=0.4, 
    ):
        super(ElectraForQuestionAnsweringAVDep2, self).__init__(config)
        
        self.num_labels = config.num_labels
        self.start_coef = start_coef
        self.end_coef = end_coef
        self.has_ans_coef = has_ans_coef
        
        self.electra = ElectraModel(config)
        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_pooler = nn.Linear(1 + config.hidden_size, 512)
        self.end_outputs = nn.Linear(512, 1)
        self.has_ans = nn.Sequential(nn.Dropout(p=config.hidden_dropout_prob), nn.Linear(config.hidden_size, 2))
        
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None, output_attentions=None,
            output_hidden_states=None,):

        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        answer_mask = attention_mask * token_type_ids
        answer_mask = answer_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility

        # batch, seq
        device = input_ids.device
        one_tensor = torch.ones((answer_mask.size(0), 1), device=device).to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        answer_mask = torch.cat([one_tensor, answer_mask[:, 1:]], dim=-1)

        start_logits = self.start_outputs(sequence_output)
        start_logits = start_logits.squeeze(-1)
        start_logits += 1000.0 * (answer_mask - 1)
        # batch, seq
        start_logits = start_logits.unsqueeze(-1)

        final_repr = gelu(self.end_pooler(torch.cat([start_logits, sequence_output], dim=-1)))
        end_logits = self.end_outputs(final_repr)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        end_logits += 1000.0 * (answer_mask - 1)

        first_word = sequence_output[:, 0, :]
        has_log = self.has_ans(first_word)

        outputs = (start_logits, end_logits, has_log, ) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(is_impossibles.size()) > 1:
                is_impossibles = is_impossibles.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            is_impossibles.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            choice_loss = loss_fct(has_log, is_impossibles)

            if self.start_coef and self.end_coef and self.has_ans_coef:
                total_loss = self.start_coef*start_loss + self.end_coef*end_loss + self.has_ans_coef*choice_loss
            else:
                total_loss = (start_loss + end_loss + self.has_ans_coef * choice_loss) / 3
            
            outputs = (total_loss,) + outputs

        
        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)