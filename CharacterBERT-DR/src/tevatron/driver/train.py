import logging
import os
import sys

import datasets
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from tevatron.arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
from tevatron.data import TrainDataset, QPCollator, TypoQPCollator, CharacterQPCollator, TypoCharacterQPCollator
from tevatron.preprocessor import HFTrainPreProcessor
from tevatron.modeling import DenseModel, DistilTypoDenseModel, DistilTypoDenseModelKD
from tevatron.trainer import DenseTrainer as Trainer, SelfTeachingDenseTrainer as TypoTrainer, GCTrainer, \
    SelfTeachingDenseTrainerKD as TypoTrainerKD


logger = logging.getLogger(__name__)


def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
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
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    if data_args.self_teaching:
        if model_args.teacher_model_name_or_path:
            model = DistilTypoDenseModelKD.build(
                model_args,
                data_args,
                training_args,
                config=config,
                cache_dir=model_args.cache_dir,
            )
        else:
            model = DistilTypoDenseModel.build(
                model_args,
                data_args,
                training_args,
                config=config,
                cache_dir=model_args.cache_dir,
            )
        if model_args.character_query_encoder:
            data_collator = TypoCharacterQPCollator(
                tokenizer,
                max_p_len=data_args.p_max_len,
                max_q_len=data_args.q_max_len
            )
        else:
            data_collator = TypoQPCollator(
                tokenizer,
                max_p_len=data_args.p_max_len,
                max_q_len=data_args.q_max_len,
                kd=True if model_args.teacher_model_name_or_path else False
            )
    else:
        model = DenseModel.build(
            model_args,
            data_args,
            training_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )

        if model_args.character_query_encoder:
            data_collator = CharacterQPCollator(
                tokenizer,
                max_p_len=data_args.p_max_len,
                max_q_len=data_args.q_max_len
            )
        else:
            data_collator = QPCollator(
                tokenizer,
                max_p_len=data_args.p_max_len,
                max_q_len=data_args.q_max_len
            )

    if data_args.train_dir is not None:
        train_dataset = TrainDataset(
            data_args,
            data_args.train_path,
            tokenizer,
            character_query_encoder=model_args.character_query_encoder,
            cache_dir=model_args.cache_dir,
            kd=True if model_args.teacher_model_name_or_path else False,
            ce_tokenzier=AutoTokenizer.from_pretrained(model_args.teacher_model_name_or_path, use_fast=False)
                            if model_args.teacher_model_name_or_path else None
        )
    else:
        train_dataset = datasets.load_dataset(data_args.dataset_name,
                                              data_args.dataset_language,
                                              cache_dir=model_args.cache_dir)[data_args.dataset_split]
        train_dataset = train_dataset.map(
            HFTrainPreProcessor(tokenizer, data_args.q_max_len, data_args.p_max_len, separator=data_args.passage_field_separator),
            batched=False,
            num_proc=data_args.dataset_proc_num,
            remove_columns=train_dataset.column_names,
            desc="Running tokenizer on train dataset",
        )
        train_dataset = TrainDataset(data_args,
                                     train_dataset,
                                     tokenizer,
                                     character_query_encoder=model_args.character_query_encoder,
                                     cache_dir=model_args.cache_dir)

    if training_args.grad_cache:
        trainer_cls = GCTrainer
    elif data_args.self_teaching:
        if model_args.teacher_model_name_or_path:
            trainer_cls = TypoTrainerKD
        else:
            trainer_cls = TypoTrainer
    else:
        trainer_cls = Trainer

    trainer = trainer_cls(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    train_dataset.trainer = trainer

    trainer.train(
        # model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
    )
    trainer.save_model(os.path.join(training_args.output_dir, f"checkpoint-final"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(os.path.join(training_args.output_dir, f"checkpoint-final"))


if __name__ == "__main__":
    main()
