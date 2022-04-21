"""Custom models for few-shot learning specific operations."""

import torch
import torch.nn as nn
import transformers
from transformers.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel, BertOnlyMLMHead
from transformers.modeling_roberta import RobertaForSequenceClassification, RobertaModel, RobertaLMHead, RobertaClassificationHead
from transformers.modeling_outputs import SequenceClassifierOutput

import logging
logger = logging.getLogger(__name__)

def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    """
    Resize the segment (token type) embeddings for BERT
    """
    if hasattr(model, 'bert'):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(new_num_types, old_token_type_embeddings.weight.size(1))
    if not random_segment:
        new_token_type_embeddings.weight.data[:old_token_type_embeddings.weight.size(0)] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types
    if hasattr(model, 'bert'):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError


class BertForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )



        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output



class RobertaForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output

class BertForPromptFinetuningWithOODDetection(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

        ##### G-ODIN code
        # h = Dense(n_classes, kernel_initializer="he_normal")(x)
        # g = Dense(1, kernel_regularizer=l2(WEIGHT_DECAY))(x)
        # g = BatchNormalization()(g)
        # g = Activation("sigmoid")(g)
        # outputs = tf.math.divide(h, g)

        self.h = nn.Linear(768, 768)
        self.g = nn.Linear(768, 1)
        self.g_bn = nn.BatchNorm1d(1)
        self.g_act = nn.Sigmoid()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )


        

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        h_seq = self.h(sequence_mask_output)

        g_seq = self.g(sequence_mask_output)
        g_seq = self.g_bn(g_seq)
        g_seq = self.g_act(g_seq)



        sequence_mask_output = torch.div(h_seq, g_seq)

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            # print(prediction_mask_scores.shape)
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output

