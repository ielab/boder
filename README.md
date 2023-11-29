# boder
Bottlenecked Pretraining for Dense Retrieval

## Installation

```bash
git clone
cd boder
pip install -r requirements.txt
```

## Examples
### Bottlenecked Pretraining on MS MARCO
```bash
export WANDB_PROJECT=BODER
RUN_NAME=msmarco_boder

python -m torch.distributed.launch --nproc_per_node 8 run.py \
  --output_dir models_pretrain_reproduce/${RUN_NAME} \
  --model_name_or_path bert-base-uncased \
  --tokenizer_name bert-base-uncased \
  --model_type boder \
  --do_train \
  --encoder_mlm_probability 0.3 \
  --decoder_mlm_probability 0.5 \
  --save_steps 20000 \
  --per_device_train_batch_size 256 \
  --max_seq_length 128 \
  --warmup_ratio 0.05 \
  --learning_rate 3e-4 \
  --max_steps 80000 \
  --overwrite_output_dir \
  --dataloader_num_workers 0 \
  --n_head_layers 2 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --report_to wandb \
  --logging_steps 100 \
  --run_name ${RUN_NAME} \
  --do_augmentation False \
  --random_mask False \
  --bottlenecked_pretrain True \
  --cache_dir cache
```

### Typos-aware Bottlenecked Pretraining (ToRoDer) on MS MARCO
For reproduce the typos robust pretraining model ToRoDer introduced in our paper [Typos-aware Bottlenecked Pre-Training for Robust Dense Retrieval](https://arxiv.org/abs/2304.08138), Shengyao Zhuang, Linjun Shou, Jian Pei, Ming Gong, Houxing Ren, Guido Zuccon and Daxin Jiang, SIGIR-AP2023. 
Simply set `--do_augmentation True` and add `--augment_probability 0.3` to the above command:

```bash
export WANDB_PROJECT=BODER
RUN_NAME=msmarco_ToRoDer

python -m torch.distributed.launch --nproc_per_node 8 run.py \
  --output_dir models_pretrain_reproduce/${RUN_NAME} \
  --model_name_or_path bert-base-uncased \
  --tokenizer_name bert-base-uncased \
  --model_type boder \
  --do_train \
  --encoder_mlm_probability 0.3 \
  --decoder_mlm_probability 0.5 \
  --save_steps 20000 \
  --per_device_train_batch_size 256 \
  --max_seq_length 128 \
  --warmup_ratio 0.05 \
  --learning_rate 3e-4 \
  --max_steps 80000 \
  --overwrite_output_dir \
  --dataloader_num_workers 0 \
  --n_head_layers 2 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --report_to wandb \
  --logging_steps 100 \
  --run_name ${RUN_NAME} \
  --do_augmentation True \
  --augment_probability 0.3 \
  --random_mask False \
  --bottlenecked_pretrain True \
  --cache_dir cache
```
For fine-tuning on MS MARCO passage ranking task with self-teaching method, please refer to our [CharacterBERT-DR repo](https://github.com/ielab/CharacterBERT-DR).

## Huggingface Checkpoints
[ielabgroup/ToRoDer](https://huggingface.co/ielabgroup/ToRoDer): Pre-trained only backbone model (Typos-aware Bottlenecked Pretrained on MS MARCO). 

[ielabgroup/ToRoDer-msmarco](https://huggingface.co/ielabgroup/ToRoDer-msmarco): Per-trained and fine-tuned (full multi-stage fine-tuning with self-teaching) on MS MARCO
