import collections
import time
import json
import paddle
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction

NUM_ANSWER = 4


@paddle.no_grad()
def evaluate(model, data_loader):
    model.eval()

    all_start_logits = []
    all_end_logits = []
    tic_eval = time.time()

    for batch in data_loader:
        input_ids, token_type_ids = batch
        start_logits_tensor, end_logits_tensor = model(input_ids,
                                                       token_type_ids)

        for idx in range(start_logits_tensor.shape[0]):
            if len(all_start_logits) % 1000 == 0 and len(all_start_logits):
                print("Processing example: %d" % len(all_start_logits))
                print('time per 1000:', time.time() - tic_eval)
                tic_eval = time.time()

            all_start_logits.append(start_logits_tensor.numpy()[idx])
            all_end_logits.append(end_logits_tensor.numpy()[idx])

    all_predictions, _, _ = compute_prediction(
        data_loader.dataset.data, data_loader.dataset.new_data,
        (all_start_logits, all_end_logits), False, 20, 30)

    # Can also write all_nbest_json and scores_diff_json files if needed
    with open('prediction.json', "w", encoding='utf-8') as writer:
        writer.write(
            json.dumps(
                all_predictions, ensure_ascii=False, indent=4) + "\n")

    squad_evaluate(
        examples=data_loader.dataset.data,
        preds=all_predictions,
        is_whitespace_splited=False)

    count = 0
    for example in data_loader.dataset.data:
        count += 1
        print()
        print('问题：', example['question'])
        print('原文：', ''.join(example['context']))
        print('答案：', all_predictions[example['id']])
        if count >= 5:
            break

    model.train()


def prepare_train_mrc_features(examples, tokenizer, doc_stride, max_seq_length):
    content, pair, label = examples
    tokenized_examples = tokenizer(pair, content, pad_to_max_seq_len=True, stride=doc_stride,
                                   max_seq_len=max_seq_length)
    # assert len(tokenized_examples) == NUM_ANSWER  # four choices
    input_ids = [tokenized_examples[i]["input_ids"] for i in range(len(tokenized_examples))]
    token_type_ids = [tokenized_examples[i]["token_type_ids"] for i in range(len(tokenized_examples))]
    return input_ids, token_type_ids, label


def prepare_train_mrc_features_i4(examples, tokenizer, doc_stride, max_seq_length):
    print('begin', len(examples))

    def _process_one(example):
        content, pair, label = example
        tokenized_examples = tokenizer(pair, content, stride=doc_stride,
                                       max_seq_len=max_seq_length)
        assert len(tokenized_examples) == NUM_ANSWER  # four choices
        ret_dict = {}
        for i in range(NUM_ANSWER):
            ret_dict[f"input_ids{i}"] = tokenized_examples[i]["input_ids"]
            ret_dict[f"token_type_ids{i}"] = tokenized_examples[i]["token_type_ids"]
        ret_dict['label'] = label
        return ret_dict

    ret_list = []

    from tqdm import tqdm
    for example in tqdm(examples):
        # ret_list.append(_process_one(example))
        d = _process_one(example)
        ret_list.append(d)
    print('finished', len(examples))
    return ret_list


def prepare_train_features(examples, tokenizer, doc_stride, max_seq_length):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    contexts = [examples[i]['context'] for i in range(len(examples))]
    questions = [examples[i]['question'] for i in range(len(examples))]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        stride=doc_stride,
        max_seq_len=max_seq_length)

    # Let's label those examples!
    for i, tokenized_example in enumerate(tokenized_examples):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_example["input_ids"]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offsets = tokenized_example['offset_mapping']

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example['token_type_ids']

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = tokenized_example['overflow_to_sample']
        answers = examples[sample_index]['answers']
        answer_starts = examples[sample_index]['answer_starts']

        # Start/end character index of the answer in the text.
        start_char = answer_starts[0]
        end_char = start_char + len(answers[0])

        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1
        # Minus one more to reach actual text
        token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and
                offsets[token_end_index][1] >= end_char):
            tokenized_examples[i]["start_positions"] = cls_index
            tokenized_examples[i]["end_positions"] = cls_index
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[
                token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples[i]["start_positions"] = token_start_index - 1
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples[i]["end_positions"] = token_end_index + 1

    return tokenized_examples


def prepare_validation_features(examples, tokenizer, doc_stride, max_seq_length):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    contexts = [examples[i]['context'] for i in range(len(examples))]
    questions = [examples[i]['question'] for i in range(len(examples))]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        stride=doc_stride,
        max_seq_len=max_seq_length)

    # For validation, there is no need to compute start and end positions
    for i, tokenized_example in enumerate(tokenized_examples):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example['token_type_ids']

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = tokenized_example['overflow_to_sample']
        tokenized_examples[i]["example_id"] = examples[sample_index]['id']

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples[i]["offset_mapping"] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_example["offset_mapping"])
        ]

    return tokenized_examples
