# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from functools import partial  # partial()函数可以用来固定某些参数值，并返回一个新的callable对象

import numpy as np
import paddle
import paddlenlp as ppnlp
import pandas as pd
from paddle import nn
from paddle.io import Dataset
from paddlenlp.data import Stack, Dict, Pad, Tuple
from paddlenlp.datasets import MapDataset, IterDataset
from sklearn.model_selection import StratifiedKFold

from mmrc.data.haihua_dataset import MyDataset, MostSimilarDataset, MostSimilarIterDataset, read_train_test
from mmrc.modeling.layers import ErnieForSequenceClassificationQ4, BertForSequenceClassificationQ4, \
    RobertaForSequenceClassificationQ4
from mmrc.utils.args_utils import ArgsParser, load_config, merge_config
from mmrc.utils.eval_utils import CVRecorder
from mmrc.utils.log_utils import get_logger
from utils import prepare_train_mrc_features, prepare_train_mrc_features_i4

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
logger = get_logger(name=__name__)
# sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

# 标准模板从yaml config 文件中读取配置，并与命令行参数合并
# standard code template for reading parameters from config file, and then merged with command line args (override)
FLAGS = ArgsParser().parse_args()
config = load_config(FLAGS.config)
merge_config(FLAGS.opt)
global_config = config['Global']

CFG = config['Global']

bert_type = global_config['bert_type']  # 0: bert, 1: erine, 2: roberta
MODEL_NAME = global_config['pretrained_model']
max_seq_length = global_config['max_len']
batch_size = global_config['train_bs']
# 训练过程中的最大学习率
learning_rate = global_config['lr']
# 训练轮次
epochs = global_config['epochs']
# 学习率预热比例
warmup_proportion = global_config['warmup_proportion']
# 权重衰减系数，类似模型正则项策略，避免模型过拟合
weight_decay = global_config['weight_decay']
random_concat = config['Global']['random_concat']
is_test_run = config['Global']['is_test_run']
dropout = None

name_tag = config['Global']['additional_tag']
support_sen_type = config['Global']['support_sen_type']
top_k = config['Global']['top_k']
fold_num = config['Global']['fold_num']
num_workers = config['Global']['num_workers']
train_file = config['Datasets']['train']
test_file = config['Datasets']['test']


def get_prefix():
    """
    :return: output path with possibly readably parameters.
    """
    prefix_formatter = f'{MODEL_NAME}_m{max_seq_length}_b{batch_size}_e{epochs}_lr{learning_rate}_d{dropout}_top' \
                       f'{top_k}_random_concat{random_concat}'
    if name_tag:
        prefix_formatter += "_" + name_tag
    return prefix_formatter


model_outputs = get_prefix()

train_df, test_df = read_train_test(train_file, test_file)

# 调用ppnlp.transformers.BertTokenizer进行数据处理，tokenizer可以把原始输入文本转化成模型model可接受的输入数据格式。
if bert_type == 1:
    tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)
elif bert_type == 0:
    tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained(MODEL_NAME)
elif bert_type == 2:
    tokenizer = ppnlp.transformers.RobertaTokenizer.from_pretrained(MODEL_NAME)

train_trans_func = partial(prepare_train_mrc_features,
                           tokenizer=tokenizer,
                           doc_stride=0,
                           max_seq_length=max_seq_length)

train_trans_func_i4 = partial(prepare_train_mrc_features_i4,
                              tokenizer=tokenizer,
                              doc_stride=0,
                              max_seq_length=max_seq_length)

# train_batch_sampler = paddle.io.DistributedBatchSampler(
#     train_ds, batch_size=batch_size, shuffle=True)
# train_batch_sampler = paddle.io.BatchSampler(
#     train_ds, batch_size=batch_size, shuffle=True)

train_batchify_fn = lambda samples, fn=Dict({
    "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
    "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    "label": Stack(dtype="int64"),
}): fn(samples)

batchify_fn = lambda samples, fn=Tuple(
    # Pad(axis=0, pad_val=vocab.get('[PAD]', 0)),  # input_ids
    Stack(dtype="int64"),
    Stack(dtype="int64"),  # seq len
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]

train_batchify_fn_i4 = lambda samples, fn=Dict({
    "input_ids0": Pad(axis=0, pad_val=tokenizer.pad_token_id),
    "input_ids1": Pad(axis=0, pad_val=tokenizer.pad_token_id),
    "input_ids2": Pad(axis=0, pad_val=tokenizer.pad_token_id),
    "input_ids3": Pad(axis=0, pad_val=tokenizer.pad_token_id),
    "token_type_ids0": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    "token_type_ids1": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    "token_type_ids2": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    "token_type_ids3": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    "label": Stack(dtype="int64"),
}): fn(samples)


def train_fn(samples):
    fn = Dict({
        "input_ids0": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "input_ids1": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "input_ids2": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "input_ids3": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids0": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        "token_type_ids1": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        "token_type_ids2": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        "token_type_ids3": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        "label": Stack(dtype="int64"),
    })
    res = fn(samples)
    return res


def train_high_level():
    if is_test_run:
        global train_df, epochs
        train_df = train_df.iloc[:50]
        epochs = 1
    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']) \
        .split(np.arange(train_df.shape[0]), train_df.label.values)  # 五折交叉验证
    only_folds = global_config['only_folds']

    best_values = []
    for fold, (trn_idx, val_idx) in enumerate(folds):
        if only_folds and fold not in only_folds:  # not empty
            continue  # skip if not empty and not in
        logger.info(f'training fold {fold} with only_fold setting: {only_folds}')
        logger.info('current val split is: len= {}, top 10 is: {}'.format(len(val_idx), val_idx))
        train = train_df.loc[trn_idx]
        val = train_df.loc[val_idx]

        if support_sen_type == 0:
            train_set = MyDataset(train)
            val_set = MyDataset(val)
        elif support_sen_type in [1, 2]:
            train_set = MostSimilarDataset(train, sel_type=support_sen_type, top_k=top_k, train=True,
                                           random_concat=random_concat)
            val_set = MostSimilarDataset(val, sel_type=support_sen_type, top_k=top_k, train=False)

        train_set = MapDataset(train_set)
        train_set.map(train_trans_func, batched=False)

        val_set = MapDataset(val_set)
        val_set.map(train_trans_func, batched=False)

        train_batch_sampler = paddle.io.BatchSampler(
            train_set, batch_size=batch_size, shuffle=True)

        train_data_loader = paddle.io.DataLoader(
            dataset=train_set,
            batch_sampler=train_batch_sampler,
            collate_fn=batchify_fn,
            num_workers=num_workers,
            # batch_size=batch_size,
            return_list=True)
        val_data_loader = paddle.io.DataLoader(
            dataset=val_set,
            collate_fn=batchify_fn,
            batch_size=batch_size,
            return_list=True)
        if bert_type == 1:
            model = ErnieForSequenceClassificationQ4.from_pretrained(MODEL_NAME)
        elif bert_type == 0:
            model = BertForSequenceClassificationQ4.from_pretrained(MODEL_NAME, dropout=dropout)
        elif bert_type == 2:
            model = RobertaForSequenceClassificationQ4.from_pretrained(MODEL_NAME)
        # model = BertForSequenceClassificationQ4(bert=pretrained)
        pmodel = paddle.Model(model)

        num_training_steps = len(train_data_loader) * epochs
        lr_scheduler = ppnlp.transformers.CosineDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            weight_decay=weight_decay)

        # Defines loss and metric.
        criterion = paddle.nn.CrossEntropyLoss()
        metric = paddle.metric.Accuracy()
        pmodel.prepare(optimizer, criterion, metric)

        earlystop = paddle.callbacks.EarlyStopping('acc', mode='max', patience=3, verbose=1, save_best_model=True)
        # Starts training and evaluating.
        logger.info('eval fold {} with {} examples'.format(fold, len(val_set)))
        pmodel.fit(train_data_loader, eval_data=val_data_loader, epochs=epochs, verbose=1,
                   save_dir=f'./{model_outputs}{fold}',
                   num_workers=10,
                   callbacks=[earlystop])
        from mmrc.utils.eval_utils import CVRecorder
        recorder = CVRecorder(fold, earlystop.best_value, train_idx=trn_idx, val_idx=val_idx, fold_num=fold_num)
        recorder.save(f'./{model_outputs}{fold}')
        # restored = CVRecorder.load(f'./{model_outputs}{fold}')
        # if restored is not None:
        #     logger.debug('pickled and restored with acc ={}'.format(restored.acc))

        best_values.append(earlystop.best_value)
        logger.info('fold {} best acc={}'.format(fold, earlystop.best_value))
        del pmodel, model, earlystop, optimizer, train_data_loader, val_data_loader
        logger.info('best values, mean={}, all={}'.format(np.mean(best_values), best_values))


def find_best_voting_strategy():
    if is_test_run:
        global train_df, epochs
        train_df = train_df.iloc[:50]
        epochs = 1
    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']) \
        .split(np.arange(train_df.shape[0]), train_df.label.values)  # 五折交叉验证
    only_folds = global_config['only_folds']
    for fold, (trn_idx, val_idx) in enumerate(folds):
        pass


def train_high_level_iter():
    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']) \
        .split(np.arange(train_df.shape[0]), train_df.label.values)  # 五折交叉验证
    only_folds = global_config['only_folds']
    for fold, (trn_idx, val_idx) in enumerate(folds):
        if only_folds and fold not in only_folds:  # not empty
            continue  # skip if not empty and not in
        train = train_df.loc[trn_idx]
        val = train_df.loc[val_idx]

        if support_sen_type == 0:
            train_set = MyDataset(train)
            val_set = MyDataset(val)
            train_set = MapDataset(train_set)
            train_set.map(train_trans_func, batched=False)
            val_set = MapDataset(val_set)
            val_set.map(train_trans_func, batched=False)
            train_batch_sampler = paddle.io.BatchSampler(
                train_set, batch_size=batch_size, shuffle=True)
        elif support_sen_type == 1:
            train_set = MostSimilarIterDataset(train, sel_type=support_sen_type, top_k=top_k,
                                               random_concat=random_concat)
            val_set = MostSimilarDataset(val, sel_type=support_sen_type, top_k=top_k)

            train_set = IterDataset(train_set)
            train_set.map(train_trans_func)
            val_set = MapDataset(val_set)
            val_set.map(train_trans_func, batched=False)
            train_batch_sampler = None

        train_data_loader = paddle.io.DataLoader(
            dataset=train_set,
            batch_sampler=train_batch_sampler,
            collate_fn=batchify_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            use_shared_memory=True,
            return_list=True)
        val_data_loader = paddle.io.DataLoader(
            dataset=val_set,
            collate_fn=batchify_fn,
            batch_size=batch_size,
            return_list=True)
        if bert_type == 1:
            model = ErnieForSequenceClassificationQ4.from_pretrained(MODEL_NAME)
        elif bert_type == 0:
            model = BertForSequenceClassificationQ4.from_pretrained(MODEL_NAME, dropout=dropout)
        elif bert_type == 2:
            model = RobertaForSequenceClassificationQ4.from_pretrained(MODEL_NAME)
        # model = BertForSequenceClassificationQ4(bert=pretrained)

        pmodel = paddle.Model(model)
        # optimizer = paddle.optimizer.Adam(parameters=pmodel.parameters(), learning_rate=learning_rate)
        # lrscheduler = paddle.optimizer.lr.LRScheduler(learning_rate=learning_rate)

        if support_sen_type == 1:
            num_training_steps = 1024 * 6 * epochs
        else:
            num_training_steps = len(train_data_loader) * epochs
        lr_scheduler = ppnlp.transformers.CosineDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            weight_decay=weight_decay)

        # Defines loss and metric.
        criterion = paddle.nn.CrossEntropyLoss()
        metric = paddle.metric.Accuracy()
        pmodel.prepare(optimizer, criterion, metric)

        earlystop = paddle.callbacks.EarlyStopping('acc', mode='max', patience=4, verbose=1, save_best_model=True)
        # paddle.callbacks.LRScheduler
        # Starts training and evaluating.
        pmodel.fit(train_data_loader, eval_data=val_data_loader, epochs=epochs, verbose=1,
                   save_dir=f'./{model_outputs}{fold}',
                   num_workers=10,
                   callbacks=[earlystop])
        del pmodel


def save(sub, values, name):
    print('value before', type(sub['label']), sub['label'].shape)
    print('value', type(values), values.shape)
    sub['label'] = values
    sub['label'] = sub['label'].apply(lambda x: ['A', 'B', 'C', 'D'][x])
    output_file = model_outputs + name + ".csv"
    sub.to_csv(output_file, index=False)


def eval_integration():
    if bert_type == 1:
        model = ErnieForSequenceClassificationQ4.from_pretrained(MODEL_NAME)
    elif bert_type == 0:
        model = BertForSequenceClassificationQ4.from_pretrained(MODEL_NAME, dropout=dropout)
    elif bert_type == 2:
        model = RobertaForSequenceClassificationQ4.from_pretrained(MODEL_NAME)

    if support_sen_type == 0:
        test_set = MyDataset(test_df)
        test_set = MapDataset(test_set)
        test_set.map(train_trans_func, batched=False)
    elif support_sen_type == 1:
        test_set = MostSimilarDataset(test_df, sel_type=support_sen_type, top_k=top_k, train=False)
        test_set = MapDataset(test_set)
        test_set.map(train_trans_func, batched=False)

    test_data_loader = paddle.io.DataLoader(
        dataset=test_set,
        collate_fn=batchify_fn,
        shuffle=False,
        batch_size=64,
        return_list=True)

    predictions = []
    predictions_norm = []
    predictions_softmax = []
    predictions_vote = []
    if is_test_run:
        CFG['fold_num'] = 2
        logger.warn('is_test_run is true')
    voting = np.zeros([CFG['fold_num'], len(test_set), 4], dtype='int32')
    accs = np.zeros((CFG['fold_num'], 1, 1), dtype='float32')

    for fold in range(CFG['fold_num']):
        pmodel = paddle.Model(model)
        print('model_output', model_outputs, name_tag)
        print('loading from', f'./{model_outputs}{fold}/best_model')
        pmodel.load(f'./{model_outputs}{fold}/best_model')
        pmodel.prepare()
        y_pred = pmodel.predict(test_data_loader)

        y_pred = np.concatenate(y_pred[0])

        # predictions_vote.append(np.argmax(y_pred, axis=1))
        for j, v in enumerate(np.argmax(y_pred, axis=1)):
            voting[fold, j, v] += 1

        predictions.append(y_pred)  # raw scores of logits
        y_pred_norm = y_pred / np.linalg.norm(y_pred, axis=-1, keepdims=True)
        predictions_norm.append(y_pred_norm)

        y_pred_exp = np.exp(y_pred)
        y_pred_softmax = y_pred_exp / np.sum(y_pred_exp, axis=1, keepdims=True)
        predictions_softmax.append(y_pred_softmax)

        restored = CVRecorder.load(f'./{model_outputs}{fold}')
        if restored is not None:
            logger.debug('restored with acc ={}'.format(restored.acc))
            accs[fold, 0, 0] = restored.acc
        else:
            logger.warn('cannot load from %s', f'./{model_outputs}{fold}')

    predictions = np.stack(predictions)
    predictions_norm = np.stack(predictions_norm)
    predictions_softmax = np.stack(predictions_softmax)

    logger.info("best acc=%s, lowest acc=%s, mean=%s", np.max(accs), np.min(accs), np.mean(accs))

    sub = pd.read_csv('sample.csv', dtype=object)  # 提交

    raw_means = np.mean(predictions, 0).argmax(1)
    raw_means_weighted = np.mean(predictions * accs, 0).argmax(1)
    save(sub, raw_means, 'raw')
    save(sub, raw_means_weighted, 'raw_weighted')

    predictions_norm_weighted = np.mean(predictions_norm * accs, 0).argmax(1)
    predictions_norm = np.mean(predictions_norm, 0).argmax(1)
    save(sub, predictions_norm, 'norm2')
    save(sub, predictions_norm_weighted, 'norm2_weighted')

    predictions_softmax_weighted = np.mean(predictions_softmax * accs, 0).argmax(1)
    predictions_softmax = np.mean(predictions_softmax, 0).argmax(1)
    save(sub, predictions_softmax, 'softmax')
    save(sub, predictions_softmax_weighted, 'softmax_weighted')

    predictions_vote = np.argmax(np.sum(voting, axis=0), axis=1)
    print(voting.shape, predictions_vote.shape)
    predictions_vote_weighted = np.argmax(np.sum(voting * accs, axis=0), axis=1)
    save(sub, predictions_vote, 'vote')
    save(sub, predictions_vote_weighted, 'vote_weighted')

    logger.info("finished and output to: %s...csv", model_outputs)


def train_tradition():
    model = BertForSequenceClassificationQ4.from_pretrained(MODEL_NAME)
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    criterion = nn.CrossEntropyLoss()

    global_step = 0
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            global_step += 1
            logits = model(*batch[:-1])
            loss = criterion(logits, batch[-1])

            if global_step % 2 == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %.5f" % (global_step, epoch, step, loss))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()


if global_config['is_train']:
    train_high_level()
    # train_high_level_iter()
if global_config['is_eval']:
    eval_integration()

# train_tradition()
