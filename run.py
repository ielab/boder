# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
# Copyright 2021 Condenser Authors
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

import logging
import math
import os
import sys
from datasets import load_dataset
from arguments import DataTrainingArguments, ModelArguments, \
    CondenserPreTrainingArguments as TrainingArguments
from data import BoderCollator, TypoBoderCollator, TrainDataset
from modeling import BoderForPretraining
from trainer import BoderPreTrainer as Trainer
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
    HfArgumentParser,
    set_seed, )
from transformers.trainer_utils import is_main_process

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        train_set = load_dataset(data_args.dataset_name, split="train", cache_dir=model_args.cache_dir)
    elif data_args.train_path is not None:
        train_set = load_dataset(
            'json',
            data_files=data_args.train_path,
            # block_size=2**25,
            cache_dir=model_args.cache_dir
        )['train']
    else:
        raise ValueError("Need to specify either dataset_name or train_path")

    dev_set = load_dataset(
        'json',
        data_files=data_args.validation_file,
        # block_size=2**25,
        cache_dir=model_args.cache_dir
    )['train'] \
        if data_args.validation_file is not None else None


    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir, use_fast=False
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=False
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    train_set = TrainDataset(train_set, tokenizer)
    dev_set = TrainDataset(dev_set, tokenizer)

    if model_args.model_name_or_path:
        # encoder = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path,
        #                                                config=config,
        #                                                cache_dir=model_args.cache_dir)
        # decoder_config = copy.deepcopy(config)
        # decoder_config.num_hidden_layers = model_args.n_head_layers
        # decoder = AutoModelForMaskedLM.from_config(config=decoder_config)
        # decoder.cls = encoder.cls
        # decoder.bert.embeddings = encoder.bert.embeddings
        # decoder.load_state_dict(_get_final_layer(2, encoder.state_dict()))
        #
        # model = BoderForPretraining(encoder, decoder, model_args, data_args, training_args)
        model = BoderForPretraining.from_pretrained(config, model_args, data_args, training_args)
    else:
        logger.warning('Training from scratch.')
        model = BoderForPretraining.from_config(
            config, model_args, data_args, training_args)

    # Data collator
    # This one will take care of randomly masking the tokens.

    if data_args.do_augmentation:
        data_collator = TypoBoderCollator(
            tokenizer=tokenizer,
            encoder_mlm_probability=data_args.encoder_mlm_probability,
            decoder_mlm_probability=data_args.decoder_mlm_probability,
            augment_probability=data_args.augment_probability,
            max_seq_length=data_args.max_seq_length,
            random_mask=data_args.random_mask
        )
    else:
        data_collator = BoderCollator(
            tokenizer=tokenizer,
            encoder_mlm_probability=data_args.encoder_mlm_probability,
            decoder_mlm_probability=data_args.decoder_mlm_probability,
            max_seq_length=data_args.max_seq_length,
            random_mask=data_args.random_mask
        )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        trainer.train(resume_from_checkpoint=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity"] = perplexity

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_mlm_wwm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in results.items():
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
