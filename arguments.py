# Copyright 2021 Condenser Author All rights reserved.
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

from dataclasses import dataclass, field
from typing import Optional, Union, List
import os
from transformers import TrainingArguments

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_path: Union[str] = field(
        default=None, metadata={"help": "Path to train data"}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train ref data file for whole word masking in Chinese."},
    )
    validation_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input validation ref data file for whole word masking in Chinese."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated. Default to the max input length of the model."
        },
    )
    min_seq_length: int = field(default=16)
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    encoder_mlm_probability: float = field(
        default=0.30, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    decoder_mlm_probability: float = field(
        default=0.50, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    augment_probability: float = field(
        default=0.10, metadata={"help": "Ratio of tokens to do augmentation."}
    )

    random_mask: bool = field(
        default=False,
        metadata={
            "help": "Whether randomly mask encoder/decoder tokens for MLM. "
        },
    )

    parallel_data: bool = field(default=False)

    train_group_size: int = field(default=8)
    dev_path: str = field(
        default=None, metadata={"help": "Path to dev data"}
    )
    pred_path: List[str] = field(default=None, metadata={"help": "Path to prediction data"})
    pred_dir: str = field(
        default=None, metadata={"help": "Path to prediction directory"}
    )
    pred_id_file: str = field(default=None)
    rank_score_path: str = field(default=None, metadata={"help": "where to save the match score"})
    max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

    do_augmentation: bool = field(
        default=False,
        metadata={
            "help": "Whether do augmentation during pretraining."
        },
    )

    def __post_init__(self):
        if self.train_dir is not None:
            files = os.listdir(self.train_dir)
            self.train_path = [
                os.path.join(self.train_dir, f)
                for f in files
                if f.endswith('tsv') or f.endswith('json') or f.endswith('jsonl')
            ]

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default='bert',
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    n_head_layers: int = field(default=2)
    bottlenecked_pretrain: bool = field(default=True)
    temperature: Optional[float] = field(default=None)


@dataclass
class CondenserPreTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    flops_weight: float = field(default=0.0)


@dataclass
class CoCondenserPreTrainingArguments(CondenserPreTrainingArguments):
    cache_chunk_size: int = field(default=-1)


@dataclass
class RerankerTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    distance_cache: bool = field(default=False)
    distance_cache_stride: int = field(default=2)

    collaborative: bool = field(default=False)