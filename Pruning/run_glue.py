# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch

import sys
sys.path.append('transformer/src/')


from configuration_bert_prun import BertConfigPrun
from modeling_bert_prun import PrunBertModel, PrunBertForSequenceClassification
from optimization_for_taylor import AdamW
from hg_transformers.configuration_auto import AutoConfig
from hg_transformers.modeling_auto import AutoModelForSequenceClassification
from hg_transformers.tokenization_auto import AutoTokenizer
from hg_transformers.trainer_utils import EvalPrediction
from hg_transformers.data.datasets.glue import GlueDataset
from hg_transformers.data.datasets.glue import GlueDataTrainingArguments as DataTrainingArguments
from hg_transformers.hf_argparser import HfArgumentParser
from hg_transformers.trainer import Trainer
from hg_transformers.training_args import TrainingArguments
from hg_transformers.trainer import Trainer
from hg_transformers.data.processors.glue import glue_output_modes
from hg_transformers.data.processors.glue import glue_tasks_num_labels
from hg_transformers.data.metrics import glue_compute_metrics
from hg_transformers.trainer import set_seed
from hg_transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule


logger = logging.getLogger(__name__)


default_params = {
        "cola": {"max_seq_length": 64, 'per_gpu_train_batch_size': 32, 'num_train_epochs': 20, 'learning_rate': 2e-5},
        "mnli": {"max_seq_length": 128, 'per_gpu_train_batch_size': 32, 'num_train_epochs': 5, 'learning_rate': 2e-5},
        "sst-2": {"max_seq_length": 64, 'per_gpu_train_batch_size': 32, 'num_train_epochs': 5, 'learning_rate': 2e-5},
        "qqp": {"max_seq_length": 128, 'per_gpu_train_batch_size': 32, 'num_train_epochs': 5, 'learning_rate': 2e-5},
        "qnli": {"max_seq_length": 128, 'per_gpu_train_batch_size': 32, 'num_train_epochs': 5, 'learning_rate': 2e-5}
    }

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_type: str = field(
        metadata={"help": "Type of the model"}
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    structured: Optional[str] = field(
        default='unstructured', metadata={"help": "whether to use structured pruning"}
    )
    is_prun: str2bool = field(
        default=True, metadata={"help": "whether to use structured pruning"}
    )
    compute_taylor: str2bool = field(
        default=False, metadata={"help": "Whether to accumulate grad to compute taylor."}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    #parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.is_prun:
        task = data_args.task_name
        training_args.per_gpu_train_batch_size = default_params[task]['per_gpu_train_batch_size']
        training_args.num_train_epochs = default_params[task]['num_train_epochs']
        training_args.learning_rate = default_params[task]['learning_rate']
        data_args.max_seq_length = default_params[task]['max_seq_length']

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
        and not model_args.compute_taylor
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.is_prun:
        config = BertConfigPrun().from_pretrained(model_args.model_name_or_path, \
                                                  num_labels=num_labels, finetuning_task=data_args.task_name)
        model = PrunBertForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)
    else:
        config = AutoConfig.from_pretrained(
            'bert-base-uncased',
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        'bert-base-uncased',
        #model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = GlueDataset(data_args, tokenizer=tokenizer) if (training_args.do_train or model_args.compute_taylor) else None
    eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode='dev') if training_args.do_eval else None
    test_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode='test') if training_args.do_predict else None

    def compute_metrics(p: EvalPrediction) -> Dict:
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        return glue_compute_metrics(data_args.task_name, preds, p.label_ids)

    if model_args.compute_taylor:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr= training_args.learning_rate, eps= training_args.adam_epsilon)
        scheduler = get_constant_schedule(optimizer)
        opt = (optimizer,scheduler)
    else:
        opt = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        optimizers=opt
    )

    fw_args = open(training_args.output_dir + '/args.txt', 'w')
    fw_args.write(str(training_args)+'\n\n')
    fw_args.write(str(model_args)+'\n\n')
    fw_args.write(str(data_args)+'\n\n')
    fw_args.close()

    # Training
    if training_args.do_train:
        _, best_score = trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode='dev'))

        for eval_dataset in eval_datasets:
            result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))
                try:
                    logger.info("  %s = %.4f", 'best_score', best_score)
                    writer.write("%s = %.4f\n" % ('best_score', best_score))
                except UnboundLocalError:
                    logger.info("This is pure evaluation.")

            results.update(result)

    if model_args.compute_taylor:
        logger.info("*** Compute First-order Taylor Expansion ***")
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        score = optimizer.get_taylor()
        score_dict = {}
        modules2prune = []
        for i in range(model.config.num_hidden_layers):
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
        output_dir = os.path.join(training_args.output_dir, 'taylor_score')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(score_dict, '%s/taylor.pkl'%output_dir)

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test")
            )

        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))

    return results



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
