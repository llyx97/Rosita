ROSITA
========
This repository contains implementation of the paper "ROSITA: Refined BERT cOmpreSsion with InTegrAted techniques
" 

The codes for fine-tuning models (w/o knowledge distillation (KD)) are modified from [huggingface/transformers](https://github.com/huggingface/transformers).

The codes for KD are modified from [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT).

The training procedure of different models is illustrated below. For more details of the experimental setup and hyper-parameters, please refer to our paper.

Requirements
========
Python3 <br />
torch=1.6.0 <br />
tqdm <br />
boto3 <br />

Data Preparation
========
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

Fine-tuning BERT-base
========
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

Training BERT(student)
========
To train the BERT(student) with the fine-tuned BERT-base as teacher, enter the directory `KD/` and run:
```
python train.py \
        --config_dir configurations/config_bert-student.json \
        --task_name ${TASK_NAME}$ \
        --do_lower_case \
        --aug_train 
```
The trained BERT(student) model will be automatically saved into `models/bert_student/${TASK_NAME}$`.

Training BERT-8layer
========
To train the BERT-8layer with BERT(student as the teacher, run:
```
python train.py \
        --config_dir configurations/config_bert-8layer.json \
        --task_name ${TASK_NAME}$ \
        --do_lower_case \
        --aug_train 
```

One-step pruning & fine-tuning
========
Step1: To compress the fine-tuned BERT-base model, enter the directory `Pruning/` and run:
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

Step2: To train the compressed model with ground-truth labels, run:
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


KD Setting1: one-step pruning + one-stage KD
========
Step1: Compress BERT as in Step1 of One-step pruning & fine-tuning.

Step2: To train the compressed model with KD Setting1, enter `KD/` and run:
```
python train.py \
        --config_dir configurations/config_setting1.json \
        --task_name ${TASK_NAME}$ \
        --do_lower_case \
        --aug_train 
```
When it comes to the augmented datasets for QNLI, QQP and MNLI, we can run `train_with_subset.py` instead, which loads the training subsets (which are constructed in Data Preparation) to reduce the memory consumption. 
