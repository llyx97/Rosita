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
The BERT(student) model will be automatically saved into `models/bert_student/${TASK_NAME}$`
