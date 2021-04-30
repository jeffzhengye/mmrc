# coding: utf-8
import ast
import json
import math

import numpy as np
import pandas as pd
import requests
from paddle.io import Dataset, IterableDataset, get_worker_info

from ..utils.log_utils import get_logger

logger = get_logger(__name__)


def similarity(texts):
    # 设置运行配置
    # 对应本地预测simnet_bow.similarity(texts=text, batch_size=1, use_gpu=True)
    data = {"texts": texts, "batch_size": 1, "use_gpu": True}
    # 指定预测方法为simnet_bow并发送post请求，content-type类型应指定json方式
    # HOST_IP为服务器IP
    url = "http://localhost:8866/predict/simnet_bow"
    headers = {"Content-Type": "application/json"}
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    return r.json()['results']


class MyDataset(Dataset):
    def __init__(self, data):
        self.df = data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):  # 将一条数据从(文章,问题,4个选项)转成(文章,问题,选项1)、(文章,问题,选项2)...
        label = self.df.label.values[idx]
        question = self.df.Question.values[idx]
        content = self.df.Content.values[idx]
        choice = self.df.Choices.values[idx]
        # if len(choice) < 4:  # 如果选项不满四个，就补“不知道”
        #     for i in range(4 - len(choice)):
        #         choice.append('D．不知道')

        content = [content for i in range(len(choice))]
        pair = [question + ' ' + i for i in choice]
        return content, pair, label


class MostSimilarDataset1(Dataset):
    def __init__(self, data, top_k=5, sel_type=0, num_choice=4):
        """
        :param data:
        :param top_k:
        :param sel_type: 0: just use original, 1: top_k sentences, 2:
        """
        self.df = data
        self.top_k = top_k
        self.sel_type = sel_type
        self.num_choice = num_choice

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):  # 将一条数据从(文章,问题,4个选项)转成(文章,问题,选项1)、(文章,问题,选项2)...
        label = self.df.label.values[idx]
        question = self.df.Question.values[idx]
        content = self.df.Content.values[idx]
        choice = self.df.Choices.values[idx]
        sentences = self.df.Sorted.values[idx]

        if self.sel_type == 1:
            content = ' '.join(sentences[:self.top_k])

        pair = [question + ' ' + i for i in choice]
        content = [content for i in range(len(choice))]

        return content, pair, label


class MostSimilarDataset(Dataset):
    def __init__(self, data, top_k=5, sel_type=0, num_choice=4, train=True, random_concat=False):
        """
        :param data:
        :param top_k:
        :param sel_type: 0: just use original, 1: top_k + first_k sentences, 2: only top_k
        """
        self.df = data
        self.origin_len = len(data)
        self.top_k = top_k
        self.sel_type = sel_type
        self.num_choice = num_choice
        self.train = train
        self.random_concat = random_concat
        logger.info("create train({}) data set with {} example (origin size={}), selection_type={}".format(train,
                                                                                                           len(self),
                                                                                                           len(self.df),
                                                                                                           sel_type))

    def __len__(self):
        if self.sel_type == 0:
            return len(self.df)
        elif self.sel_type == 1 and self.train:
            return len(self.df) * 2
        elif self.sel_type == 1 and not self.train:
            return len(self.df)
        elif self.sel_type == 2:  #
            return len(self.df)

    def __getitem__(self, index):  # 将一条数据从(文章,问题,4个选项)转成(文章,问题,选项1)、(文章,问题,选项2)...
        idx = index if index < self.origin_len else index - self.origin_len
        label = self.df.label.values[idx]
        question = self.df.Question.values[idx]
        content = self.df.Content.values[idx]
        choice = self.df.Choices.values[idx]
        sentences = self.df.Sorted.values[idx]

        if self.sel_type == 2:
            content = ' '.join(sentences[:self.top_k])
        elif index >= self.origin_len and self.sel_type == 1 or not self.train:
            content = ' '.join(sentences[:self.top_k])

        if self.random_concat and self.train and np.random.uniform(0, 1) > 0.5:
            pair = [i + ' ' + question for i in choice]
        else:
            pair = [question + ' ' + i for i in choice]
        content = [content for i in range(len(choice))]

        return content, pair, label


class MostSimilarIterDataset(IterableDataset):
    def __init__(self, data, top_k=5, sel_type=0, num_choice=4):
        """
        :param data:
        :param top_k:
        :param sel_type: 0: just use original, 1: top_k sentences, 2:
        """
        # self.df = data
        data_list = data.to_dict('list')
        self.keys = data_list.keys()
        self.questions, self.choices, _, self.contents, self.sentences_list, self.labels = data_list.values()
        self.top_k = top_k
        self.sel_type = sel_type
        self.num_choice = num_choice
        self.start = 0
        self.end = len(self.questions)
        print('total num', self.end)

    def __iter__(self):  # 将一条数据从(文章,问题,4个选项)转成(文章,问题,选项1)、(文章,问题,选项2)...
        worker_info = get_worker_info()
        if worker_info is None:
            iter_start = self.start
            iter_end = self.end
        else:
            per_worker = int(
                math.ceil((self.end - self.start) / float(
                    worker_info.num_workers)))
            #             print('per_worker', per_worker)
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        print("worker info:", worker_info.id, iter_start, iter_end)

        try:
            # cur_df = self.df.iloc[iter_start: iter_end]
            # print('keys', cur_df.to_dict('list').keys())
            # for r in zip(*cur_df.to_dict('list').values()):
            # print('r len', len(r))
            # question, choice, _, content, sentences, label = r
            # for _, row in self.df.iloc[iter_start: iter_end].iterrows():
            for idx in range(iter_start, iter_end):
                label = self.labels[idx]
                question = self.questions[idx]
                content = self.contents[idx]
                choice = self.choices[idx]
                sentences = self.sentences_list[idx]

                # label = self.df.label.values[idx]
                # question = self.df.Question.values[idx]
                # content = self.df.Content.values[idx]
                # choice = self.df.Choices.values[idx]
                # sentences = self.df.Sorted.values[idx]
                # label = row['label']
                # question = row['Question']
                # content = row['Content']
                # choice = row['Choices']
                # sentences = row['Sorted']
                pair = [question + ' ' + c for c in choice]
                content = [content] * self.num_choice
                yield content, pair, label

                if self.sel_type == 1:
                    content = ' '.join(sentences[:self.top_k])
                    # print('selected content len:', worker_info.id, idx, len(content), len(sentences))
                    content = [content] * self.num_choice
                    yield content, pair, label
        except Exception as e:
            logger.error(str(e), exc_info=True)
            exit()


def read_train_test(train_file, test_file):
    def choice_completion(choice):
        choice = [i[2:] for i in choice]
        if len(choice) < 4:  # 如果选项不满四个，就补“不知道”
            for i in range(4 - len(choice)):
                choice.append('不知道')
        return choice

    usecols = ['Question', 'Choices', 'Content', 'Sorted', 'label', 'Answer']
    converters = {'Choices': ast.literal_eval, 'Sorted': ast.literal_eval}
    train_df = pd.read_csv(train_file, converters=converters, usecols=usecols)
    test_df = pd.read_csv(test_file, converters=converters, usecols=usecols[:-1])

    train_df['Choices'] = train_df['Choices'].apply(choice_completion)
    test_df['Choices'] = test_df['Choices'].apply(choice_completion)
    train_df['label'] = train_df['Answer'].apply(lambda x: ['A', 'B', 'C', 'D'].index(x))  # 将标签从ABCD转成0123
    test_df['label'] = 0
    return train_df, test_df


# train_df = train_df.iloc[:200]


# def read(data_path, df_source):
#     for i, row in df_source.iterrows():
#         yield {'questions': row['Question'], 'choices': row['Choices'], 'labels': row['label']}
#
#
# import inspect
#
# print(inspect.isfunction(read))
# print(type(partial(read, df_source=train_df)))
#
# # data_path为read()方法的参数
# train_ds = load_dataset(partial(read, df_source=train_df), data_path=train_file_path, lazy=False)
# # iter_ds = load_dataset(partial(read, df_source=train_df), data_path=train_file_path, lazy=True)
# test_ds = load_dataset(partial(read, df_source=test_df), data_path=train_file_path, lazy=False)


# train_ds = MyDataset(train_df)
# test_ds = MyDataset(test_df)

if __name__ == "__main__":
    pass
