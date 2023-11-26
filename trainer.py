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

import os
from contextlib import nullcontext

from typing import Dict, List, Tuple, Optional, Any, Union

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch import nn, Tensor
from torch.cuda.amp import autocast
from torch.utils.data import DistributedSampler, Dataset
from transformers.trainer import Trainer
try:
    from grad_cache import GradCache
    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False

from torch.utils.data import DataLoader, RandomSampler
from torch.utils.checkpoint import get_device_states, set_device_states
from torch.utils.data.distributed import DistributedSampler
from transformers.trainer import Trainer, nested_detach
from transformers.trainer_utils import PredictionOutput, EvalPrediction
from modeling import Reranker, RerankerDC
import logging

logger = logging.getLogger(__name__)


class BoderPreTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(BoderPreTrainer, self).__init__(*args, **kwargs)
        self._encoder_loss = None
        self._decoder_loss = None
        self._contrastive_loss = None

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save_pretrained'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save_pretrained interface')
        else:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _remove_unused_columns(self, dataset, description: Optional[str] = None):
        # we are not going to do this in this
        # as collator will be generating new columns
        pass

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.args.warmup_ratio > 0:
            self.args.warmup_steps = num_training_steps * self.args.warmup_ratio

        super().create_optimizer_and_scheduler(num_training_steps)

    def compute_loss(self, model, inputs):
        encoder_inputs, decoder_inputs = inputs
        encoder_labels = encoder_inputs.pop('labels')
        decoder_labels = decoder_inputs.pop('labels')

        encoder_loss, decoder_loss, contrastive_loss = model(encoder_inputs, encoder_labels, decoder_inputs, decoder_labels)

        # for logging different losses
        # if self._encoder_loss is None:
        #     self._encoder_loss = torch.tensor(0.0).to(self.args.device)
        # if self._decoder_loss is None:
        #     self._decoder_loss = torch.tensor(0.0).to(self.args.device)
        # if self._contrastive_loss is None:
        #     self._contrastive_loss = torch.tensor(0.0).to(self.args.device)
        # self._encoder_loss += encoder_loss
        # self._decoder_loss += decoder_loss
        # self._contrastive_loss += contrastive_loss
        # if self.state.global_step != 0 and (self.state.global_step+1) % self.args.logging_steps == 0:
        #     encoder_loss_scalar = self._nested_gather(self._encoder_loss).mean().item()
        #     decoder_loss_scalar = self._nested_gather(self._decoder_loss).mean().item()
        #     contrastive_loss_scalar = self._nested_gather(self._contrastive_loss).mean().item()
        #     # reset _loss to zero
        #     self._encoder_loss -= self._encoder_loss
        #     self._decoder_loss -= self._decoder_loss
        #     self._contrastive_loss -= self._contrastive_loss
        #     logs: Dict[str, float] = {}
        #     logs["encoder_loss"] = round(encoder_loss_scalar / ((self.state.global_step+1) - self._globalstep_last_logged), 4)
        #     logs["decoder_loss"] = round(decoder_loss_scalar / ((self.state.global_step+1) - self._globalstep_last_logged), 4)
        #     logs["contrastive_loss"] = round(
        #         contrastive_loss_scalar / ((self.state.global_step + 1) - self._globalstep_last_logged), 4)
        #     self.log(logs)

        return encoder_loss + decoder_loss + contrastive_loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        encoder_inputs, decoder_inputs = inputs
        encoder_labels = encoder_inputs.pop('labels')
        decoder_labels = decoder_inputs.pop('labels')


        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if self.args.fp16:
                with autocast():
                    encoder_loss, decoder_loss, contrastive_loss = model(encoder_inputs, encoder_labels, decoder_inputs,
                                                                         decoder_labels)
            else:
                encoder_loss, decoder_loss, contrastive_loss = model(encoder_inputs, encoder_labels, decoder_inputs,
                                                                     decoder_labels)

            loss = encoder_loss + decoder_loss + contrastive_loss

        return (loss, None, None)


class ParallelPreTrainer(BoderPreTrainer):
    def __init__(self, *args, **kwargs):
        super(ParallelPreTrainer, self).__init__(*args, **kwargs)
        self._mlm_loss = None
        self._flops_loss = None
        self._expansion_loss = None

    def compute_loss(self, model, inputs):
        encoder_inputs1, encoder_inputs2 = inputs
        encoder_labels1 = encoder_inputs1.pop('labels')
        encoder_labels2 = encoder_inputs2.pop('labels')

        mlm_loss, flops_loss, expansion_loss = model(encoder_inputs1,
                                                     encoder_labels1,
                                                     encoder_inputs2,
                                                     encoder_labels2)

        # for logging different losses
        if self._mlm_loss is None:
            self._mlm_loss = torch.tensor(0.0).to(self.args.device)
        if self._flops_loss is None:
            self._flops_loss = torch.tensor(0.0).to(self.args.device)
        if self._expansion_loss is None:
            self._expansion_loss = torch.tensor(0.0).to(self.args.device)
        self._mlm_loss += mlm_loss
        self._flops_loss += flops_loss
        self._expansion_loss += expansion_loss
        if self.state.global_step != 0 and (self.state.global_step+1) % self.args.logging_steps == 0:
            mlm_loss_scalar = self._nested_gather(self._mlm_loss).mean().item()
            flops_loss_scalar = self._nested_gather(self._flops_loss).mean().item()
            expansion_loss_scalar = self._nested_gather(self._expansion_loss).mean().item()
            # reset _loss to zero
            self._mlm_loss -= self._mlm_loss
            self._flops_loss -= self._flops_loss
            self._expansion_loss -= self._expansion_loss
            logs: Dict[str, float] = {}
            logs["mlm_loss"] = round(mlm_loss_scalar / ((self.state.global_step+1) - self._globalstep_last_logged), 4)
            logs["flops_loss"] = round(flops_loss_scalar / ((self.state.global_step+1) - self._globalstep_last_logged), 4)
            logs["expansion_loss"] = round(
                expansion_loss_scalar / ((self.state.global_step + 1) - self._globalstep_last_logged), 4)
            self.log(logs)

        flops_loss = self.args.flops_weight * flops_loss
        mlm_loss = 0.1 * mlm_loss
        return mlm_loss + flops_loss + expansion_loss


class SyncedSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0) -> None:
        super(SyncedSampler, self).__init__(
            dataset, num_replicas, rank, shuffle, seed)
        self.num_samples = len(self.dataset)
        self.total_size = len(self.dataset)

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        # DO NOT SUB SAMPLE!
        assert len(indices) == self.total_size
        assert len(indices) == self.num_samples

        return iter(indices)

    def set_epoch(self, epoch: int):
        super(SyncedSampler, self).set_epoch(epoch)
        logger.info(f'Setting Data Sampler Epoch to {epoch}')

class RerankerTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save_pretrained'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save_pretrained interface')
        else:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _get_train_sampler(self):
        if self.args.local_rank == -1:
            return RandomSampler(self.train_dataset)
        elif self.args.collaborative:
            logger.info(f'Collaborative Mode.')
            return SyncedSampler(self.train_dataset, seed=self.args.seed)
        else:
            return DistributedSampler(self.train_dataset)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.args.warmup_ratio > 0:
            self.args.warmup_steps = num_training_steps * self.args.warmup_ratio

        return super(RerankerTrainer, self).create_optimizer_and_scheduler(num_training_steps)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        Will use no sampler if :obj:`self.train_dataset` is a :obj:`torch.utils.data.IterableDataset`, a random sampler
        (adapted to distributed training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model: Reranker, inputs):
        return model(inputs)['loss']

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if self.args.fp16:
                with autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

            loss = None
            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
            else:
                logits = outputs

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        labels = None

        return (loss, logits, labels)

    def prediction_loop(
            self,
            *args,
            **kwargs
    ) -> PredictionOutput:
        pred_outs = super().prediction_loop(*args, **kwargs)
        preds, label_ids, metrics = pred_outs.predictions, pred_outs.label_ids, pred_outs.metrics
        preds = preds.squeeze()
        if self.compute_metrics is not None:
            metrics_no_label = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics_no_label = {}

        for key in list(metrics_no_label.keys()):
            if not key.startswith("eval_"):
                metrics_no_label[f"eval_{key}"] = metrics_no_label.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics={**metrics, **metrics_no_label})

class RandContext:
    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(
            devices=self.fwd_gpu_devices,
            enabled=True
        )
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None

class RerankerDCTrainer(RerankerTrainer):
    def _chunk_input(self, inputs: Dict[str, torch.Tensor], chunk_size: int = None):
        if chunk_size is None:
            chunk_size = self.args.distance_cache_stride
        keys = list(inputs.keys())
        for k, v in inputs.items():
            inputs[k] = v.split(chunk_size)

        chunks = []
        n_chunks = len(inputs[keys[0]])

        for i in range(n_chunks):
            chunks.append({k: inputs[k][i] for k in keys})

        return chunks

    def training_step(self, model: RerankerDC, inputs):
        model.train()
        _model = getattr(model, 'module', model)
        inputs = self._prepare_inputs(inputs)

        rnd_states = []
        all_logits = []
        chunks = self._chunk_input(inputs)

        for chunk in chunks:
            rnd_states.append(RandContext())
            if self.args.fp16:
                with torch.no_grad():
                    with autocast():
                        chunk_logits = model(chunk)
            else:
                with torch.no_grad():
                    chunk_logits = model(chunk)
            all_logits.append(chunk_logits)

        all_logits = torch.cat(all_logits).float()
        loss, grads = _model.compute_grad(all_logits)
        grads = grads.view(-1, self.args.distance_cache_stride)

        for chunk_id, chunk in enumerate(chunks):
            with rnd_states[chunk_id]:
                if self.args.fp16:
                    with autocast():
                        surrogate = model(chunk, grads[chunk_id])
                else:
                    surrogate = model(chunk, grads[chunk_id])

            if self.args.gradient_accumulation_steps > 1:
                surrogate = surrogate / self.args.gradient_accumulation_steps

            if self.args.fp16:
                self.scaler.scale(surrogate).backward()
            else:
                surrogate.backward()

        return loss.detach()
