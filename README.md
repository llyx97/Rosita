# ROSITA

This repository contains implementation of the paper "ROSITA: Refined BERT cOmpreSsion with InTegrAted techniques" in AAAI 2021.

The code for fine-tuning models (w/o knowledge distillation (KD)) is modified from [huggingface/transformers](https://github.com/huggingface/transformers).

The code for KD is modified from [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT).

The training procedure of different models is illustrated as follows

## Requirements

Python3 <br />
torch=1.6.0 <br />
tqdm <br />
boto3 <br />

## Data Preparation

Step1: Download the [GLUE data](https://gluebenchmark.com/tasks) by running:
```
python download_glue_data.py --data_dir data --tasks all
```
The extracted .tsv files for CoLA and SST-2 are given in the `data/` folder.

Step2: Download the pre-trained language model BERT (bert-base-uncased) and [GloVe](http://nlp.stanford.edu/data) (glove.42B.300d) embeddings, to `models/bert_pt/` and `glove/` respectively. The download of BERT can be achieved by running:
```
python download_bert.py
```

Step3: Conduct data augmentation by running:
```
python data_augmentation.py --pretrained_bert_model models/bert_pt \
                            --glove_embs glove/glove.42B.300d.txt \
                            --glue_dir data \  
                            --task_name ${TASK_NAME}$
```
The augmented dataset `train_aug.tsv` will be automatically saved into `data/${TASK_NAME}$`.

Step4 (Optional): For QNLI, QQP and MNLI which have millions of augmented data, we divide `train_aug.tsv` into subsets to reduce the memory consumption in training. 

The following (Linux) commands can be used to split the dataset:
```
cd data/${TASK_NAME}$
split -500000 -d train_aug.tsv train_aug
```
Now we have subsets with 500,000 data samples in each. Rename the subset files as `train_aug0.tsv, train_aug1.tsv ...`

## Fine-tuning BERT-base

To fine-tune the pre-trained BERT model on a downstream task ${TASK_NAME}$, enter the directory `Pruning/` and run:
```
python run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --task_name ${TASK_NAME}$ \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --data_dir ../data/${TASK_NAME}$ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir ../models/bert_ft/${TASK_NAME}$ \
  --logging_dir ../models/bert_ft/${TASK_NAME}$/logging \
  --logging_steps 50 \
  --save_steps 0 \
  --is_prun False \
```

## Training BERT(student)

To train the BERT(student) with the fine-tuned BERT-base as teacher, enter the directory `KD/` and run:
```
python train.py \
        --config_dir configurations/config_bert-student.json \
        --task_name ${TASK_NAME}$ \
        --do_lower_case \
        --aug_train 
```
The trained BERT(student) model will be automatically saved into `models/bert_student/${TASK_NAME}$`.

## Training BERT-8layer

To train the BERT-8layer with BERT(student) as the teacher, run:
```
python train.py \
        --config_dir configurations/config_bert-8layer.json \
        --task_name ${TASK_NAME}$ \
        --do_lower_case \
        --aug_train 
```
In this training process, the top-most layers of BERT(student) will be iteratively compressed until the number of layers reduces to 8.

## One-step pruning + fine-tuning w/o KD

Requirements: [Fine-tuned BERT](https://github.com/llyx97/Rosita#fine-tuning-bert-base)

Step1: To compress the fine-tuned BERT-base model, we first need to determine the importance of model weights. We use a metric based on first-order taylor expansion, which can be computed by entering the directory `Pruning/` and runnning:
```
python run_glue.py \
  --model_type bert \
  --task_name ${TASK_NAME}$ \
  --data_dir ../data/${TASK_NAME}$ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --num_train_epochs 1.0 \
  --save_steps 0 \
  --model_name_or_path ../models/bert_ft/${TASK_NAME}$ \
  --output_dir ../models/bert_ft/${TASK_NAME}$ \
  --compute_taylor True \
  --is_prun False 
```
The results will be saved to `../models/bert_ft/${TASK_NAME}$/taylor_score/taylor.pkl`

Step2: Now we can conduct model compression by running:
```
python3 pruning_one_step.py \
        -model_path ../models/bert_ft \
        -output_dir ../models/prun_bert \
        -task ${TASK_NAME}$ \
        -keep_heads ${NUM_OF_ATTN_HEADS_TO_KEEP}$ \
        -num_layers ${NUM_OF_LAYERS_TO_KEEP}$ \
        -ffn_hidden_dim ${HIDDEN_DIM_OF_FFN}$ \
        -emb_hidden_dim ${MATRIX_RANK_OF_EMB_FACTORIZATION}$
```
The four hyperparameters `keep_heads`, `keep_layers`, `ffn_hidden_dim` and `emb_hidden_dim` construct a space of the model architecture.
In the final setting of ROSITA, `keep_heads=2`, `keep_layers=8`, `ffn_hidden_dim=512` and `emb_hidden_dim=128`.

Step3: To train the compressed model with ground-truth labels, run:
```
python run_glue.py \
  --model_type bert \
  --model_name_or_path ../models/prun_bert/CoLA/a2_l8_f512_e128 \
  --task_name CoLA \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --data_dir ../data/CoLA \
  --max_seq_length 64 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 20.0 \
  --output_dir ../models/one_step_prun/CoLA/a2_l8_f512_e128 \
  --logging_dir ../models/one_step_prun/CoLA/a2_l8_f512_e128/logging \
  --logging_steps 100 \
  --save_steps 0 \
  --is_prun True
```
where the training hyperparameter are for the CoLA dataset. For settings of the other datasets, please refer to the appendix of our paper.


## Fine-tuning ROSITA w/ KD


### KD Setting1: one-step pruning + one-stage KD

Requirements: [Fine-tuned BERT](https://github.com/llyx97/Rosita#fine-tuning-bert-base)

Step1: Compress BERT as in Step1 and Step2 of One-step pruning + fine-tuning w/o KD.

Step2: To train the compressed model with KD Setting1, enter `KD/` and run:
```
python train.py \
        --config_dir configurations/config_setting1.json \
        --task_name ${TASK_NAME}$ \
        --do_lower_case \
        --aug_train 
```
When it comes to the augmented datasets for QNLI, QQP and MNLI, we can run `train_with_subset.py` instead, which loads the training subsets (constructed in Data Preparation) to reduce the memory consumption. `train_with_subset.py` can also be used in the other KD settings.


KD Setting2: one-step pruning + two-stage KD
========
Requirements: [Fine-tuned BERT](https://github.com/llyx97/Rosita#fine-tuning-bert-base), [BERT(student)](https://github.com/llyx97/Rosita#training-bertstudent)

Step1: Compute the weight importance metric by entering `KD/` and running:
```
python train.py \
        --teacher_model ../models/bert_ft/${TASK_NAME}$ \
        --student_model ../models/bert_student/${TASK_NAME}$ \
        --data_dir ../data/${TASK_NAME}$ \
        --task_name ${TASK_NAME}$ \
        --output_dir ../models/bert_student/${TASK_NAME}$/taylor_score \
        --num_train_epochs 1 \
        --do_lower_case \
        --pred_distill \
        --compute_taylor 
```

Step2: Compress the BERT(student) model by running:
```
python3 pruning_one_step.py \
        -model_path ../models/bert_student \
        -output_dir ../models/prun_bert_student \
        -task ${TASK_NAME}$ \
        -keep_heads 2 \
        -num_layers 8 \
        -ffn_hidden_dim 512 \
        -emb_hidden_dim 128
```

Step3: Train the compressed model by running:
```
python train.py \
        --config_dir configurations/config_setting2.json \
        --task_name ${TASK_NAME}$ \
        --do_lower_case \
        --aug_train
```


KD Setting3: iterative width pruning + two-stage KD
========
Requirements: [BERT(student)](https://github.com/llyx97/Rosita#training-bertstudent)

Step1: Compress BERT(student) to 8 layers by entering `KD/` and running:
```
python3 pruning_one_step.py \
        -model_path ../models/bert_student \
        -output_dir ../models/bert-8layer/one_step_prun \
        -task ${TASK_NAME}$ \
        -keep_heads 12 \
        -num_layers 8 \
        -ffn_hidden_dim 3072 \
        -emb_hidden_dim -1
```

Step2: Train and iteratively compress the 8layer BERT model by running:
```
python train.py \
        --config_dir configurations/config_setting3.json \
        --task_name ${TASK_NAME}$ \
        --do_lower_case \
        --aug_train
```


KD Setting4: iterative width&depth pruning + three-stage KD
========
Requirements: [BERT-8layer](https://github.com/llyx97/Rosita#training-bert-8layer) trained with iterative depth pruning

Train and iteratively compress the BERT-8layer model by entering `KD/` and running:
```
python train.py \
        --config_dir configurations/config_setting4.json \
        --task_name ${TASK_NAME}$ \
        --do_lower_case \
        --aug_train
```

Evaluation
========
To evaluate any trained model on the dev and test sets, enter `KD/` and run:
```
python test.py \
        --student_model ${PATH_OF_THE_MODEL_TO_EVALUATE}$ \
        --output_dir  ${PATH_TO_OUTPUT_EVALUATION_RESULTS}$ \
        --data_dir ../data/${TASK_NAME}$ \
        --task_name ${TASK_NAME}$ \
        --do_lower_case \
        --do_eval \
        --do_predict
```
Here we provide the ROSITA models trained under KD Setting4 for evaluation:

[ROSITA for CoLA](https://drive.google.com/file/d/1ZHpQ5p_xJuHDKT2uCuxYbgN9vbibVt8o/view?usp=sharing)

[ROSITA for SST-2](https://drive.google.com/file/d/1zUhs-6GEzgZH7sD7_bQIPrke_MEo-1KT/view?usp=sharing)

[ROSITA for QNLI](https://drive.google.com/file/d/1ICS31Et2zIrbt2znXKCO9VpT9qhR49b7/view?usp=sharing)

[ROSITA for QQP](https://drive.google.com/file/d/1ijDkd9uQVtvh3hIv3fXajEmqonmk56zX/view?usp=sharing)

[ROSITA for MNLI](https://drive.google.com/file/d/1R06Ie80CUj9UxKFceQ2dM4xLOxpw9w8B/view?usp=sharing)

Citation
========
If you use this repository in a published research, please cite our paper:
```
@inproceedings{ROSITA,
author = {Yuanxin Liu and Zheng Lin and Fengcheng Yuan},
title = {ROSITA: Refined BERT cOmpreSsion with InTegrAted techniques},
booktitle = {AAAI 2021},
year = {2021}
}
```
