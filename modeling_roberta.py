
# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import math
import torch
import torch.nn as nn

from transformers import *
from torch.nn import CrossEntropyLoss, MSELoss
from modules import SCAttention, split_ques_context, TrmCoAtt
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class RobertaForQuestionAnsweringAVPool(RobertaPreTrainedModel):
    def __init__(self, config, no_answer_loss_coef:float=1.0, ):
        super(RobertaForQuestionAnsweringAVPool, self).__init__(config)
        self.num_labels = config.num_labels
        self.no_answer_loss_coef = no_answer_loss_coef
        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.has_ans = nn.Sequential(nn.Dropout(p=config.hidden_dropout_prob), nn.Linear(config.hidden_size, 2))
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
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


class RobertaForQuestionAnsweringAVPoolBCE(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaForQuestionAnsweringAVPoolBCE, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.has_ans = nn.Sequential(nn.Dropout(p=config.hidden_dropout_prob), nn.Linear(config.hidden_size, 1))
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None):

        outputs = self.roberta(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        first_word = sequence_output[:, 0, :]

        # has_inp = torch.cat([p_avg, first_word, q_summ, p_avg_p], -1)
        has_log = self.has_ans(first_word)
        has_log = has_log.squeeze(-1)
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
            
            choice_fct = nn.BCEWithLogitsLoss()

            is_impossibles = is_impossibles.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility

            choice_loss = choice_fct(has_log, is_impossibles)
            total_loss = (start_loss + end_loss + choice_loss) / 3
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

class RobertaForQuestionAnsweringSeqSC(RobertaPreTrainedModel):
    def __init__(self, config, no_answer_loss_coef: float=1.0, ):
        super(RobertaForQuestionAnsweringSeqSC, self).__init__(config)
        self.num_labels = config.num_labels
        self.no_answer_loss_coef = no_answer_loss_coef
        self.roberta = RobertaModel(config)
        self.attention = SCAttention(config.hidden_size, config.hidden_size)
        self.has_ans = nn.Sequential(nn.Dropout(p=config.hidden_dropout_prob), nn.Linear(config.hidden_size, 2))
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, pq_end_pos=None, 
                position_ids=None, head_mask=None, inputs_embeds=None, start_positions=None, 
                end_positions=None, is_impossibles=None
    ):

        outputs = self.roberta(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        query_sequence_output, context_sequence_output, query_attention_mask, context_attention_mask = \
            split_ques_context(sequence_output, pq_end_pos)
        sequence_output = self.attention(sequence_output, query_sequence_output, query_attention_mask)

        sequence_output = sequence_output + outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        first_word = sequence_output[:, 0, :]
        has_log = self.has_ans(first_word)

        outputs = (start_logits, end_logits,) + outputs[2:]
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

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

class RobertaForQuestionAnsweringSeqTrm(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaForQuestionAnsweringSeqTrm, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.albert_att = TrmCoAtt(config)
        self.has_ans = nn.Sequential(nn.Dropout(p=config.hidden_dropout_prob), nn.Linear(config.hidden_size, 2))
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, pq_end_pos=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None):

        outputs = self.roberta(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        query_sequence_output, context_sequence_output, query_attention_mask, context_attention_mask = \
            split_ques_context(sequence_output, pq_end_pos)

        sequence_output = self.albert_att(query_sequence_output, sequence_output, query_attention_mask)

        sequence_output = sequence_output + outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        first_word = sequence_output[:, 0, :]
        has_log = self.has_ans(first_word)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            choice_loss = loss_fct(has_log, is_impossibles)
            total_loss = (start_loss + end_loss + choice_loss) / 3
            # total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class RobertaForQuestionAnsweringAVPoolBCEV3(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaForQuestionAnsweringAVPoolBCEV3, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.has_ans1 = nn.Sequential(nn.Dropout(p=config.hidden_dropout_prob), nn.Linear(config.hidden_size, 2))
        self.has_ans2 = nn.Sequential(nn.Dropout(p=config.hidden_dropout_prob), nn.Linear(config.hidden_size, 1))
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None):

        outputs = self.roberta(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        first_word = sequence_output[:, 0, :]

        # has_inp = torch.cat([p_avg, first_word, q_summ, p_avg_p], -1)
        has_log1 = self.has_ans1(first_word)
        has_log2 = self.has_ans2(first_word)
        has_log1 = has_log1.squeeze(-1)
        has_log2 = has_log2.squeeze(-1)

        outputs = (start_logits, end_logits, has_log1,) + outputs[2:]

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
            
            choice_fct = nn.BCEWithLogitsLoss()

            choice_loss1 = loss_fct(has_log1, is_impossibles)

            is_impossibles = is_impossibles.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility

            choice_loss2 = choice_fct(has_log2, is_impossibles)

            loss_fct = MSELoss()
            choice_loss3 = loss_fct(has_log2.view(-1), is_impossibles.view(-1))
            
            total_loss = (start_loss + end_loss + choice_loss1 + choice_loss2 + choice_loss3) / 5
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class RobertaForQuestionAnsweringAVDep(RobertaPreTrainedModel):
    def __init__(
        self, 
        config, 
        start_coef: float=None, 
        end_coef: float=None,
        has_ans_coef: float=None, 
    ):
        super(RobertaForQuestionAnsweringAVDep, self).__init__(config)
        
        self.num_labels = config.num_labels
        self.start_coef = start_coef
        self.end_coef = end_coef
        self.has_ans_coef = has_ans_coef
        
        self.roberta = RobertaModel(config)
        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_pooler = nn.Linear(1 + config.hidden_size, 512)
        self.end_outputs = nn.Linear(512, 1)
        self.has_ans = nn.Sequential(nn.Dropout(p=config.hidden_dropout_prob), nn.Linear(config.hidden_size, 2))
        
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
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


class RobertaForBinaryClassification(BertPreTrainedModel):
   config_class = RobertaConfig
   base_model_prefix = "roberta"
   def __init__(self, config):
        super(RobertaForBinaryClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(4*config.hidden_size, self.num_labels)

        self.init_weights()

   def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):
                
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               # token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        cls_output = torch.cat((
            outputs[2][-1][:,0, ...], 
            outputs[2][-2][:,0, ...], 
            outputs[2][-3][:,0, ...], 
            outputs[2][-4][:,0, ...]),- 1
        )
        logits = self.qa_outputs(cls_output)
        
        return logits

class RobertaForQuestionAnsweringAVDep2(RobertaPreTrainedModel):
    def __init__(
        self, 
        config, 
        start_coef: float=None, 
        end_coef: float=None,
        has_ans_coef: float=None, 
    ):
        super(RobertaForQuestionAnsweringAVDep2, self).__init__(config)
        
        self.num_labels = config.num_labels
        self.start_coef = start_coef
        self.end_coef = end_coef
        self.has_ans_coef = has_ans_coef
        
        self.roberta = RobertaModel(config)
        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_pooler = nn.Linear(1 + config.hidden_size, 512)
        self.end_outputs = nn.Linear(512, 1)
        self.has_ans = nn.Sequential(nn.Dropout(p=config.hidden_dropout_prob), nn.Linear(config.hidden_size, 2))
        
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
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
                total_loss = (start_loss + end_loss + choice_loss) / 3
            
            outputs = (total_loss,) + outputs

        
        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)