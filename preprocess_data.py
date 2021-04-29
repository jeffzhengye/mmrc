# encoding: utf-8
import json
import pandas as pd
from ltp import LTP
import paddlehub as hub
import paddlenlp as ppnlp
import paddle
import numpy as np
import requests

import json

# CUDA_VISIBLE_DEVICES=2 hub serving start -m simnet_bow  # how to start a service.
ltp = LTP()
# simnet_bow = hub.Module(name="simnet_bow")

train_file_path = "/mnt/d/datasets/nlp/mrc/haihua/train.json"
validation_file_path = "/mnt/d/datasets/nlp/mrc/haihua/validation.json"
train_file_path = "/home/nieshuang/public/haihua/train.json"
validation_file_path = "/home/nieshuang/public/haihua/validation.json"

content = u'我是中国人'
question = u'我贵'
options = [u'答案1', u'苹果', u'香蕉', u'不知道']
MODEL_NAME = 'roberta-wwm-ext'
tokenizer = ppnlp.transformers.RobertaTokenizer.from_pretrained(MODEL_NAME)
model = ppnlp.transformers.RobertaModel.from_pretrained(MODEL_NAME)
model.eval()


def get_inputs_type_ids(text):
    def _process_list(texts):
        max_seq_len = max([len(t) for t in texts]) + 1
        # print('max_seq_len', max_seq_len)
        tokenized_examples = tokenizer(texts, pad_to_max_seq_len=True, max_seq_len=max_seq_len)
        input_ids = [tokenized_examples[i]["input_ids"] for i in range(len(tokenized_examples))]
        token_type_ids = [tokenized_examples[i]["token_type_ids"] for i in range(len(tokenized_examples))]
        return input_ids, token_type_ids

    if isinstance(text, str):
        return _process_list([text])
    elif isinstance(text, list):
        return _process_list(text)
    else:
        raise NotImplementedError()


def model_eval(text):
    input_ids, token_type_ids = get_inputs_type_ids(text)

    input_ids = paddle.to_tensor(input_ids, dtype='int64')
    token_type_ids = paddle.to_tensor(token_type_ids, dtype='int64')

    outputs, outputs1 = model(input_ids=input_ids, token_type_ids=token_type_ids)
    return outputs, outputs1


# print(_)

def get_sorted_sentences(d):
    content = d['Content']
    question = d['Question']
    choices = d['Choices']
    sentences = LTP.sent_split([content])
    scores_list = []
    for i in range(len(choices)):
        text2 = [question + choices[i][2:]] * len(sentences)
        results = simnet_bow.similarity(texts=[sentences, text2])
        scores = [result['similarity'] for result in results]
        scores_list.append(scores)
    means = np.mean(scores_list, axis=0)
    index = np.argsort(means)[::-1]
    sorted_sentences = [sentences[i] for i in index]
    return sorted_sentences


def get_cosine_sorted_sentences(d):
    content = d['Content']
    question = d['Question']
    choices = d['Choices']
    sentences = LTP.sent_split([content])
    scores_list = np.zeros(len(sentences), dtype='float64')
    question_out, _ = model_eval(question)  # [1, token_num, hidden_size]
    options_out, _ = model_eval(choices)  # [1, options_num, token_num, hidden_size]
    question_out_l2 = question_out / paddle.norm(question_out, p=2, axis=-1, keepdim=True)
    options_out_l2 = options_out / paddle.norm(options_out, p=2, axis=-1, keepdim=True)

    for i, sentence in enumerate(sentences):
        content_out, _ = model_eval(sentence)  # [1, token_num, hidden_size]
        content_out_l2 = content_out / paddle.norm(content_out, p=2, axis=-1, keepdim=True)

        qc = paddle.matmul(question_out_l2, content_out_l2, transpose_y=True)
        qc_pooled = paddle.mean(paddle.max(qc, axis=-1))

        oc = paddle.matmul(options_out_l2, content_out_l2, transpose_y=True)  # [1, options_num, token_num, token_num]
        oc_pooled = paddle.mean(paddle.max(oc, axis=-1))
        scores_list[i] = qc_pooled.numpy() + oc_pooled.numpy()

    index = np.argsort(scores_list)[::-1]
    sorted_sentences = [sentences[i] for i in index]
    return sorted_sentences


def get_sorted_cosine(d):
    # import paddlehub as hub
    # lac = hub.Module(name="lac")
    # test_text = ["今天是个好日子", "天气预报说今天要下雨", "下一班地铁马上就要到了"]
    # results = lac.cut(text=test_text, use_gpu=False, batch_size=1, return_tag=True)

    # print(type(results))
    # for result in results:
    #     print(result['word'])
    #     # print(result['tag'])
    pass


def sorted_sentences():
    with open(train_file_path, 'r', encoding='utf-8')as f:  # 读入json文件
        train_data = json.load(f)
    train_df = []
    from tqdm import tqdm

    for i in tqdm(range(len(train_data)), ncols=80):  # 将每个文章-问题-答案作为一条数据
        # for i in tqdm(range(10)):
        data = train_data[i]
        content = data['Content']
        questions = data['Questions']
        for question in questions:
            question['Content'] = content
            sorted_sentences = get_sorted_sentences(question)
            question['Sorted'] = sorted_sentences
            train_df.append(question)

    # print(train_df[0])

    train_df = pd.DataFrame(train_df)  # 转换成csv表格更好看一点

    # print(train_df.head())

    with open(validation_file_path, 'r', encoding='utf-8')as f:
        test_data = json.load(f)

    test_df = []

    for i in range(len(test_data), ncols=80):
        # for i in tqdm(range(10)):
        data = test_data[i]
        content = data['Content']
        questions = data['Questions']
        cls = data['Type']
        diff = data['Diff']
        for question in questions:
            question['Content'] = content
            question['Type'] = cls
            question['Diff'] = diff
            sorted_sentences = get_sorted_sentences(question)
            question['Sorted'] = sorted_sentences
            test_df.append(question)

    test_df = pd.DataFrame(test_df)

    train_df['content_len'] = train_df['Content'].apply(len)  # 统计content文本长度
    test_df['content_len'] = test_df['Content'].apply(len)

    # print(train_df.content_len.describe())
    # print(test_df.content_len.describe())

    # plt.title('content_length') # content非常长，绝大部分都远大于512
    # plt.plot(sorted(train_df.content_len))
    # plt.show()

    train_df['label'] = train_df['Answer'].apply(lambda x: ['A', 'B', 'C', 'D'].index(x))  # 将标签从ABCD转成0123
    test_df['label'] = 0

    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    print('finished')


def sorted_sentences_cosine(input_file, is_train=True, sorted_fun=get_sorted_sentences, new_column='Sorted'):
    with open(input_file, 'r', encoding='utf-8')as f:  # 读入json文件
        train_data = json.load(f)

    from tqdm import tqdm
    train_df = []
    for i in tqdm(range(len(train_data)), ncols=80):  # 将每个文章-问题-答案作为一条数据
        data = train_data[i]
        content = data['Content']
        questions = data['Questions']
        for question in questions:
            question['Content'] = content
            sorted_sentences = sorted_fun(question)
            question[new_column] = sorted_sentences
            train_df.append(question)

    train_df = pd.DataFrame(train_df)  # 转换成csv表格更好看一点
    train_df['content_len'] = train_df['Content'].apply(len)  # 统计content文本长度

    if is_train:
        train_df['label'] = train_df['Answer'].apply(lambda x: ['A', 'B', 'C', 'D'].index(x))  # 将标签从ABCD转成0123
    else:
        train_df['label'] = 0
    import os
    basename = os.path.basename(input_file)
    suffix = '_cosine'
    output_name = basename + suffix + ".csv"
    train_df.to_csv(output_name, index=False)
    print('finished processing', input_file)


sorted_sentences_cosine(input_file=train_file_path, is_train=True, sorted_fun=get_cosine_sorted_sentences)
sorted_sentences_cosine(input_file=validation_file_path, is_train=False, sorted_fun=get_cosine_sorted_sentences)
