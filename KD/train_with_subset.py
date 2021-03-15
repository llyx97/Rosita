# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team and Huawei Noah's Ark Lab.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse, csv, logging, os, random, sys, json, torch, gc
import numpy as np
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from transformer.modeling import TinyBertForSequenceClassification
from transformer.modeling_prun import TinyBertForSequenceClassification as PrunTinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME

csv.field_size_limit(sys.maxsize)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

try:
    from tensorboardX import SummaryWriter
    _has_tensorboard = True
except ImportError:
    _has_tensorboard = False
def is_tensorboard_available():
    return _has_tensorboard


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir, subset_id):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug%d.tsv"%subset_id)), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_aug_examples(self, data_dir, subset_id):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug%d.tsv"%subset_id)), "aug")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir, subset_id):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug%d.tsv"%subset_id)), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir, subset_id):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug%d.tsv"%subset_id)), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir, subset_id):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug%d.tsv"%subset_id)), "aug")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir, subset_id):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug%d.tsv"%subset_id)), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_aug_examples(self, data_dir, subset_id):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug%d.tsv"%subset_id)), "aug")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir, subset_id):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug%d.tsv"%subset_id)), "aug")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir, subset_id):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug%d.tsv"%subset_id)), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(example.label))
            logger.info("label_id: {}".format(label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          seq_length=seq_length))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids


def result_to_file(result, file_name):
    with open(file_name, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write('\n')


def do_eval(model, task_name, eval_dataloader,
            device, output_mode, eval_labels, num_labels):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_

            logits, _, _ = model(input_ids, segment_ids, input_mask)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, eval_labels.numpy())
    result['eval_loss'] = eval_loss
    return result



def pruning(model, optimizer, model_path, keep_layers, keep_heads, ffn_hidden_dim, emb_hidden_dim, \
        num_labels, prun_step, device, task, schedule, next_lr, next_t_total):
    score = optimizer.get_taylor(prun_step)
    score_dict = {}
    modules2prune = []
    for i in range(12):
        modules2prune += ['encoder.layer.%d.attention.self.query.weight'%i,
                         'encoder.layer.%d.attention.self.key.weight'%i,
                         'encoder.layer.%d.attention.self.value.weight'%i,
                         'encoder.layer.%d.attention.output.dense.weight'%i,
                         'encoder.layer.%d.intermediate.dense.weight'%i,
                         'encoder.layer.%d.output.dense.weight'%i]
    for name, param in list(model.bert.named_parameters()):
        if name in modules2prune:
            cur_score = score[param]
            score_dict[name] = cur_score
    output_dir = os.path.join(model_path, 'temp')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(score_dict, os.path.join(output_dir, 'taylor.pkl'))

    prun_command = "python3 pruning.py -model_path %s\
                                    -keep_heads %d\
                                    -num_layers %d\
                                    -ffn_hidden_dim %d\
                                    -emb_hidden_dim %d\
                                    -task %s\
                                    "%(model_path, keep_heads, keep_layers, ffn_hidden_dim, emb_hidden_dim, task)
    os.system(prun_command)
    model = PrunTinyBertForSequenceClassification.from_pretrained(output_dir, num_labels=num_labels)
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                             schedule=schedule,
                             lr=next_lr,
                             warmup=0.,
                             t_total=next_t_total,
                             device=device)
    return model, optimizer



def iterative_pruning(args, student_model, teacher_model, optimizer, tokenizer,
                        num_train_optimization_steps, prun_step, max_prun_times,
                        prun_times, num_labels, device):

    model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
    model_name = WEIGHTS_NAME
    output_dir = os.path.join(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_model_file = os.path.join(output_dir,  model_name)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)

    if args.depth_or_width == 'width':
        keep_heads, ffn_hidden_dim, emb_hidden_dim = pruning_schedule(prun_times, max_prun_times,
            teacher_model.config.num_attention_heads, teacher_model.config.prun_intermediate_size,
            teacher_model.config.hidden_size,
            args.ffn_hidden_dim, args.emb_hidden_dim)
        keep_layers = student_model.config.num_hidden_layers
    elif args.depth_or_width == 'depth':
        keep_heads, ffn_hidden_dim, emb_hidden_dim = args.keep_heads, args.ffn_hidden_dim, args.emb_hidden_dim
        keep_layers = teacher_model.config.num_hidden_layers-prun_times

    next_t_total = num_train_optimization_steps-prun_times*prun_step
    if args.lr_schedule=='none':
        next_lr = args.learning_rate
    elif args.lr_schedule=='warmup_linear':
        next_lr = optimizer.param_groups[0]['lr'] * optimizer.param_groups[0]['schedule'].get_lr(prun_step)
    student_model, optimizer = pruning(student_model, optimizer, args.output_dir, 
                        keep_layers, keep_heads, ffn_hidden_dim, emb_hidden_dim, num_labels, 
                        prun_step, device, args.task_name, args.lr_schedule, next_lr, next_t_total)
    return student_model, optimizer



def pruning_schedule(prun_times, max_prun_times,
        orig_heads, orig_ffn_dim, orig_emb_dim,
        final_ffn_dim, final_emb_dim):
    keep_heads = orig_heads - prun_times
    ffn_hidden_dim = orig_ffn_dim - (orig_ffn_dim-final_ffn_dim)*(prun_times/max_prun_times)
    emb_hidden_dim = orig_emb_dim - (orig_emb_dim-final_emb_dim)*(prun_times/max_prun_times)
    if final_emb_dim==-1:
        emb_hidden_dim = -1
    return keep_heads, ffn_hidden_dim, emb_hidden_dim



def init_prun(args, tlayer_num, tatt_heads, num_train_examples):
    if args.depth_or_width=='depth':
        max_prun_times, prun_times = tlayer_num - args.keep_layers, 0
    elif args.depth_or_width=='width':
        max_prun_times, prun_times = tatt_heads - args.keep_heads, 0
    prun_freq = args.prun_period_proportion * args.num_train_epochs / max_prun_times
    prun_step = int(prun_freq * num_train_examples/args.train_batch_size/args.gradient_accumulation_steps)
    return max_prun_times, prun_times, prun_freq, prun_step



def build_dataloader(set_type, args, processor, label_list, tokenizer, output_mode, subset_id=None):
    if set_type=='train' and args.aug_train:
        examples = processor.get_aug_examples(args.data_dir, subset_id)
    elif set_type=='train':
        examples = processor.get_train_examples(args.data_dir)
    elif set_type=='eval':
        examples = processor.get_dev_examples(args.data_dir)
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode)
    data, labels = get_tensor_data(output_mode, features)
    sampler = SequentialSampler(data) if set_type=='eval' else RandomSampler(data)
    batch_size = args.eval_batch_size if set_type=='eval' else args.train_batch_size
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader, labels, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        help="The teacher model dir.")
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        help="The student model dir.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--config_dir",
                        default=None,
                        type=str,
                        help="The directory of configuration file.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--lr_schedule",
                        default="constant",
                        type=str,
                        help="Learning rate schedule.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--patience",
                        default=None,
                        type=int,
                        help="The max number of epoch without improvement. Used to perform early stop.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--prun_period_proportion',
                        type=float,
                        default=0.,
                        help="The proportion of pruning steps.")
    parser.add_argument('--keep_heads',
                        type=int,
                        default=None,
                        help="Number of attention heads to keep.")
    parser.add_argument('--ffn_hidden_dim',
                        type=int,
                        default=None,
                        help="Number of FFN hhidden dimension to keep.")
    parser.add_argument('--emb_hidden_dim',
                        type=int,
                        default=None,
                        help="Number of embedding hidden dim to keep.")
    parser.add_argument('--keep_layers',
                        type=int,
                        default=None,
                        help="Number of layers to keep.")

    # added arguments
    parser.add_argument('--aug_train',
                        action='store_true')
    parser.add_argument('--eval_step',
                        type=int,
                        default=50)
    parser.add_argument('--pred_distill',
                        action='store_true')
    parser.add_argument('--repr_distill',
                        action='store_true')
    parser.add_argument('--data_url',
                        type=str,
                        default="")
    parser.add_argument('--temperature',
                        type=float,
                        default=1.)

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor
    }

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification",
        "wnli": "classification"
    }

    # intermediate distillation default parameters
    default_params = {
            "cola": {"max_seq_length": 64, 'train_batch_size': 32},
            "mnli": {"max_seq_length": 128, 'train_batch_size': 64},
            "mrpc": {"max_seq_length": 128, 'train_batch_size': 32},
            "sst-2": {"max_seq_length": 64, 'train_batch_size': 32},
            "sts-b": {"max_seq_length": 128, 'train_batch_size': 32},
            "qqp": {"max_seq_length": 128, 'train_batch_size': 64},
            "qnli": {"max_seq_length": 128, 'train_batch_size': 64},
            "rte": {"max_seq_length": 128, 'train_batch_size': 32}
    }

    # number of augmented data for each dataset
    num_aug_data = {
            "cola": 213080,
            "mnli": 8049121,
            "qnli": 4246837,
            "sst-2": 1118455,
            "qqp": 7621089
    }

    # number of subsets for each dataset
    num_subsets = {
            "mnli": 17,
            "qnli": 9,
            "qqp":4
    }

    acc_tasks = ["mnli", "mrpc", "sst-2", "qqp", "qnli", "rte"]
    corr_tasks = ["sts-b"]
    mcc_tasks = ["cola"]

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # Prepare seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    task_name = args.task_name.lower()
    if not args.do_eval:
        config_file = open(args.config_dir, 'r')
        config = json.load(config_file)
        config_file.close()
        for key in config[task_name]:
            vars(args)[key] = config[task_name][key]
        args.logging_dir = os.path.join(args.output_dir, 'logging')
        if not os.path.exists(args.logging_dir):
            os.makedirs(args.logging_dir)
        if is_tensorboard_available():
            tb_writer = SummaryWriter(log_dir=args.logging_dir)
    assert args.aug_train==True, "Use subset training only when data augmentation is applied."

    # Prepare task settings
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if task_name in default_params:
        args.max_seq_length = default_params[task_name]["max_seq_length"]
        args.train_batch_size = default_params[task_name]["train_batch_size"]

    fw_args = open(args.output_dir + '/args.txt', 'w')
    fw_args.write(str(args))
    fw_args.close()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    tokenizer = BertTokenizer.from_pretrained(args.student_model, do_lower_case=args.do_lower_case)

    if not args.do_eval:
        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))
        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        num_train_examples = num_aug_data[task_name]
        num_train_optimization_steps = int(
            num_train_examples / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    eval_dataloader, eval_labels, eval_data = build_dataloader('eval', args, processor, label_list, tokenizer, output_mode)

    if not args.do_eval:
        teacher_model = TinyBertForSequenceClassification.from_pretrained(args.teacher_model, num_labels=num_labels)
        teacher_model.to(device)
    student_model = PrunTinyBertForSequenceClassification.from_pretrained(args.student_model, num_labels=num_labels)
    student_model.to(device)
    
    if args.do_eval:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_data))
        logger.info("  Batch size = %d", args.eval_batch_size)

        student_model.eval()
        result = do_eval(student_model, task_name, eval_dataloader,
                         device, output_mode, eval_labels, num_labels)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    else:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", num_train_examples)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        if n_gpu > 1:
            student_model = torch.nn.DataParallel(student_model)
            teacher_model = torch.nn.DataParallel(teacher_model)
        # Prepare optimizer
        param_optimizer = list(student_model.named_parameters())
        size = 0
        for n, p in student_model.named_parameters():
            logger.info('n: {}'.format(n))
            size += p.nelement()

        logger.info('Total parameters: {}'.format(size))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             schedule=args.lr_schedule,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps,
                             device=device)
        # Prepare loss functions
        loss_mse = MSELoss()
        def soft_cross_entropy(predicts, targets):
            student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
            targets_prob = torch.nn.functional.softmax(targets, dim=-1)
            return (- targets_prob * student_likelihood).mean()

        # Train and evaluate
        global_step = 0
        best_dev_acc = 0.0
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        if 'depth_or_width' in args:
            max_prun_times, prun_times, prun_freq, prun_step = init_prun(args, teacher_model.config.num_hidden_layers, 
                                                            teacher_model.config.num_attention_heads, len(train_data))
            best_acc_step = max_prun_times * prun_step
        else:
            prun_times, max_prun_times, prun_step = -1, -1, -1
            best_acc_step = 0

        for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0.
            tr_rep_loss = 0.
            tr_cls_loss = 0.

            student_model.train()
            nb_tr_examples, nb_tr_steps, epoch_step = 0, 0, 0

            for subset_id in range(num_subsets[task_name]):
                if not (subset_id==0 and epoch_==0):
                    try:
                        del train_examples, train_features, train_data, train_sampler, train_dataloader
                    except NameError:
                        del train_data, train_sampler, train_dataloader
                    gc.collect()
                try:
                    train_data = torch.load(os.path.join(args.data_dir, 'train_aug%d.pt'%subset_id))
                    train_sampler = RandomSampler(train_data)
                    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
                except FileNotFoundError:
                    train_dataloader, _, train_data = build_dataloader('train', args, processor, label_list, tokenizer, output_mode, subset_id)
                    torch.save(train_data, os.path.join(args.data_dir, 'train_aug%d.pt'%subset_id))

                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                    batch = tuple(t.to(device) for t in batch)

                    input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
                    if input_ids.size()[0] != args.train_batch_size:
                        continue
                    if (args.patience is not None) and (global_step-best_acc_step >= args.patience*num_train_optimization_steps/args.num_train_epochs):
                        return

                    if (global_step+1)%prun_step == 0 and prun_times<max_prun_times and 'depth_or_width' in args:
                        prun_times += 1
                        logger.info("Pruning after %.2f epoches"%(prun_times*prun_freq))
                        student_model, optimizer = iterative_pruning(args, student_model, teacher_model, optimizer, tokenizer,
                                                                    num_train_optimization_steps, prun_step, max_prun_times, 
                                                                    prun_times, num_labels, device)
                    rep_loss = 0.
                    cls_loss = 0.

                    student_logits, _, student_reps = student_model(input_ids, segment_ids, input_mask,
                                                                               is_student=True)
                    with torch.no_grad():
                        teacher_logits, _, teacher_reps = teacher_model(input_ids, segment_ids, input_mask)

                    if args.repr_distill:
                        teacher_layer_num = teacher_model.config.num_hidden_layers
                        student_layer_num = student_model.config.num_hidden_layers
                        prun_layer_ratio = student_layer_num/teacher_layer_num
                        layers_per_block = int(teacher_layer_num / student_layer_num)

                        try:
                            assert teacher_layer_num % student_layer_num == 0
                            new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                        except AssertionError:
                            # Use mod to drop layers if not teacher_layer_num % student_layer_num == 0
                            new_teacher_reps = [teacher_reps[i] for i in range(teacher_layer_num+1) if not (i+1)%(1/prun_layer_ratio)<1e-5]

                        for student_rep, teacher_rep in zip(student_reps, new_teacher_reps):
                            tmp_loss = loss_mse(student_rep, teacher_rep)
                            rep_loss += tmp_loss
                        tr_rep_loss += rep_loss.item()

                    if args.pred_distill:
                        if output_mode == "classification":
                            cls_loss = soft_cross_entropy(student_logits / args.temperature,
                                                          teacher_logits / args.temperature)
                        elif output_mode == "regression":
                            loss_mse = MSELoss()
                            cls_loss = loss_mse(student_logits.view(-1), label_ids.view(-1))

                        tr_cls_loss += cls_loss.item()
                    loss = rep_loss + cls_loss

                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    loss.backward()
                    optimizer.accumulate_grad()

                    tr_loss += loss.item()
                    nb_tr_examples += label_ids.size(0)
                    nb_tr_steps += 1

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1
                        epoch_step += 1

                    if (global_step + 1) % args.eval_step == 0:
                        logger.info("***** Running evaluation *****")
                        logger.info("  Epoch = {} iter {} step".format(epoch_, global_step))
                        logger.info("  Num examples = %d", len(eval_data))
                        logger.info("  Batch size = %d", args.eval_batch_size)
                        lr = optimizer.param_groups[0]['lr'] * \
                        optimizer.param_groups[0]['schedule'].get_lr(global_step-prun_step*prun_times)
                        for group in optimizer.param_groups:
                            for p in group['params']:
                                optim_step = optimizer.state[p]['step']
                                break
                            break
                        logger.info("  Learning rate = %.2e", lr)

                        student_model.eval()

                        loss = tr_loss / (epoch_step + 1)
                        cls_loss = tr_cls_loss / (epoch_step + 1)
                        rep_loss = tr_rep_loss / (epoch_step + 1)

                        result = {}
                        if args.pred_distill:
                            result = do_eval(student_model, task_name, eval_dataloader,
                                             device, output_mode, eval_labels, num_labels)
                        result['global_step'] = global_step
                        result['cls_loss'] = cls_loss
                        result['rep_loss'] = rep_loss
                        result['loss'] = loss
                        result['lr'] = lr
                        result['optim_step'] = optim_step
                        if tb_writer is not None:
                            for k, v in result.items():
                                tb_writer.add_scalar(k, v, global_step)


                        if (not args.pred_distill) or (prun_times<max_prun_times):
                            save_model = True
                        elif  prun_times>=max_prun_times:
                            save_model = False

                            if task_name in acc_tasks and result['acc'] > best_dev_acc:
                                best_dev_acc = result['acc']
                                save_model = True
                                best_acc_step = global_step

                            if task_name in corr_tasks and result['corr'] > best_dev_acc:
                                best_dev_acc = result['corr']
                                save_model = True
                                best_acc_step = global_step

                            if task_name in mcc_tasks and result['mcc'] > best_dev_acc:
                                best_dev_acc = result['mcc']
                                save_model = True
                                best_acc_step = global_step


                        if save_model:
                            logger.info("***** Save model *****")
                            result_to_file(result, output_eval_file)

                            model_to_save = student_model.module if hasattr(student_model, 'module') else student_model

                            model_name = WEIGHTS_NAME
                            # if not args.pred_distill:
                            #     model_name = "step_{}_{}".format(global_step, WEIGHTS_NAME)
                            output_model_file = os.path.join(args.output_dir, model_name)
                            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                            tokenizer.save_vocabulary(args.output_dir)

                            # Test mnli-mm
                            #if args.pred_distill and task_name == "mnli":
                            #    task_name = "mnli-mm"
                            #    processor = processors[task_name]()
                            #    if not os.path.exists(args.output_dir + '-MM'):
                            #        os.makedirs(args.output_dir + '-MM')

                            #    eval_examples = processor.get_dev_examples(args.data_dir)

                            #    eval_features = convert_examples_to_features(
                            #        eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
                            #    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)

                            #    logger.info("***** Running mm evaluation *****")
                            #    logger.info("  Num examples = %d", len(eval_examples))
                            #    logger.info("  Batch size = %d", args.eval_batch_size)

                            #    eval_sampler = SequentialSampler(eval_data)
                            #    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                            #                                 batch_size=args.eval_batch_size)

                            #    result = do_eval(student_model, task_name, eval_dataloader,
                            #                     device, output_mode, eval_labels, num_labels)

                            #    result['global_step'] = global_step

                            #    tmp_output_eval_file = os.path.join(args.output_dir + '-MM', "eval_results.txt")
                            #    result_to_file(result, tmp_output_eval_file)

                            #    task_name = 'mnli'
                        student_model.train()
        if 'tb_writer' in dir():
            tb_writer.close()

if __name__ == "__main__":
    main()
