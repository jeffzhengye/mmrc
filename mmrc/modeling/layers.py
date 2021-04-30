# -*- coding: utf-8 -*-
'''
  @author: jeffzhengye
  @contact: yezheng@scuec.edu.cn
  @file: layers.py
  @time: 2021/2/2 21:52
  @desc:
 '''
from paddlenlp.transformers import BertPretrainedModel, ErniePretrainedModel, RobertaPretrainedModel
from paddle import nn
import paddlenlp as ppnlp
import paddle



class BertForSequenceClassification(BertPretrainedModel):
    """
    Model for sentence (pair) classification task with BERT.
    Args:
        bert (BertModel): An instance of BertModel.
        num_classes (int, optional): The number of classes. Default 2
        dropout (float, optional): The dropout probability for output of BERT.
            If None, use the same value as `hidden_dropout_prob` of `BertModel`
            instance `bert`. Default None
    """

    def __init__(self, bert, num_classes=1, dropout=None):
        super(BertForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.bert = bert  # allow bert to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.bert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.bert.config["hidden_size"],
                                    num_classes)
        self.flatten = nn.Flatten(start_axis=1)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        _, pooled_output = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        # pooled_output = paddle.reshape(pooled_output, [-1, 4, pooled_output.shape[-1]])

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = paddle.reshape(logits, [-1, 4])
        # logits = self.flatten(logits)
        return logits


class BertForSequenceClassificationQ4(BertPretrainedModel):
    def __init__(self, bert, num_classes=1, num_choice=4, dropout=None):
        super(BertForSequenceClassificationQ4, self).__init__()
        self.num_classes = num_classes
        self.num_choice = num_choice
        self.bert = bert  # allow bert to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.bert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.bert.config["hidden_size"],
                                    num_classes)
        self.flatten = nn.Flatten(start_axis=1)
        self.apply(self.init_weights)

    def forward1(self, inputs0, inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7):
        position_ids = None
        attention_mask = None
        input_ids = [inputs0, inputs1, inputs2, inputs3, ]
        token_type_ids = [inputs4, inputs5, inputs6, inputs7, ]

        pooled_outputs = []
        for i in range(len(input_ids)):
            _, pooled_output = self.bert(
                input_ids[i],
                token_type_ids=token_type_ids[i],
                position_ids=position_ids,
                attention_mask=attention_mask)
            pooled_outputs.append(pooled_output)

        # pooled_output = paddle.reshape(pooled_output, [-1, 4, pooled_output.shape[-1]])

        pooled_outputs = [self.dropout(pooled_output) for pooled_output in pooled_outputs]

        logits = [self.classifier(pooled_output) for pooled_output in pooled_outputs]
        logits = paddle.concat(logits, axis=1)
        return logits

    def forward(self, inputs0, inputs1):
        position_ids = None
        attention_mask = None
        input_ids = paddle.reshape(inputs0, [-1, inputs0.shape[-1]])
        token_type_ids = paddle.reshape(inputs1, [-1, inputs1.shape[-1]])

        _, pooled_outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)
        pooled_outputs = paddle.reshape(pooled_outputs, [-1, self.num_choice, pooled_outputs.shape[-1]])

        pooled_outputs = self.dropout(pooled_outputs)
        # pooled_outputs = [self.dropout(pooled_output) for pooled_output in pooled_outputs]
        #
        logits = self.classifier(pooled_outputs)
        logits = self.flatten(logits)
        # return logits
        return logits


# class ErnieForQuestionAnswering(ErniePretrainedModel):
#     def __init__(self, ernie):
#         super(ErnieForQuestionAnswering, self).__init__()
#         self.ernie = ernie  # allow ernie to be config
#         self.classifier = nn.Linear(self.ernie.config["hidden_size"], 2)
#         self.apply(self.init_weights)
#
#     def forward(self,
#                 input_ids,
#                 token_type_ids=None,
#                 position_ids=None,
#                 attention_mask=None):
#         sequence_output, _ = self.ernie(
#             input_ids,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             attention_mask=attention_mask)
#
#         logits = self.classifier(sequence_output)
#         logits = paddle.transpose(logits, perm=[2, 0, 1])
#         start_logits, end_logits = paddle.unstack(x=logits, axis=0)
#
#         return start_logits, end_logits


class ErnieForSequenceClassificationQ4(ErniePretrainedModel):
    def __init__(self, ernie, num_classes=1, num_choice=4, dropout=None):
        super(ErnieForSequenceClassificationQ4, self).__init__()
        self.num_classes = num_classes
        self.num_choice = num_choice
        self.ernie = ernie  # allow bert to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ernie.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"],
                                    num_classes)
        self.flatten = nn.Flatten(start_axis=1)
        self.apply(self.init_weights)

    def forward(self, inputs0, inputs1):
        position_ids = None
        attention_mask = None
        input_ids = paddle.reshape(inputs0, [-1, inputs0.shape[-1]])
        token_type_ids = paddle.reshape(inputs1, [-1, inputs1.shape[-1]])

        _, pooled_outputs = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)
        pooled_outputs = paddle.reshape(pooled_outputs, [-1, self.num_choice, pooled_outputs.shape[-1]])

        pooled_outputs = self.dropout(pooled_outputs)
        logits = self.classifier(pooled_outputs)
        logits = self.flatten(logits)
        return logits


class RobertaForSequenceClassificationQ4(RobertaPretrainedModel):
    def __init__(self, roberta, num_classes=1, num_choice=4, dropout=None):
        super(RobertaForSequenceClassificationQ4, self).__init__()
        self.num_classes = num_classes
        self.num_choice = num_choice
        self.roberta = roberta  # allow bert to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.roberta.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.roberta.config["hidden_size"],
                                    num_classes)
        self.flatten = nn.Flatten(start_axis=1)
        self.apply(self.init_weights)

    def forward(self, inputs0, inputs1):
        position_ids = None
        attention_mask = None
        input_ids = paddle.reshape(inputs0, [-1, inputs0.shape[-1]])
        token_type_ids = paddle.reshape(inputs1, [-1, inputs1.shape[-1]])

        _, pooled_outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)
        pooled_outputs = paddle.reshape(pooled_outputs, [-1, self.num_choice, pooled_outputs.shape[-1]])

        pooled_outputs = self.dropout(pooled_outputs)
        logits = self.classifier(pooled_outputs)
        logits = self.flatten(logits)
        return logits
