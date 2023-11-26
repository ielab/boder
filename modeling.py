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
import copy
import os
import random
import warnings
from typing import Dict, Optional
import torch
from torch import nn, Tensor
from torch.cuda.amp import autocast
import collections
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
import torch.nn.functional as F
from transformers import BertModel, BertConfig, AutoModel, AutoModelForMaskedLM, AutoConfig, PretrainedConfig, \
    RobertaModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer,\
    PreTrainedModel, PreTrainedTokenizer

from transformers.models.bert.modeling_bert import BertPooler, BertOnlyMLMHead, BertPreTrainingHeads, BertLayer
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPooling, MaskedLMOutput
from transformers.models.roberta.modeling_roberta import RobertaLayer

from arguments import DataTrainingArguments, ModelArguments, CoCondenserPreTrainingArguments
from transformers import TrainingArguments
import logging

logger = logging.getLogger(__name__)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _get_final_layer(num, model):
    state = collections.OrderedDict()

    for k, v in model.items():
        if 'encoder.layer' in k:
            index = k.find('encoder.layer.') + len('encoder.layer.')
            prefix, layer_num = k[:index], k[index:]
            layer_num, suffix = int(layer_num[:layer_num.find('.')]), layer_num[layer_num.find('.'):]
            if 12 - layer_num <= num:
                new_k = '%s%d%s' % (prefix, layer_num - 12 + num, suffix)
                state[new_k] = v
        else:
            state[k] = v

    return state


class BoderForPretraining(nn.Module):
    _keys_to_ignore_on_save = None
    def __init__(
            self,
            encoder,
            decoder,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            training_args: TrainingArguments
    ):
        super(BoderForPretraining, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cross_entropy = nn.CrossEntropyLoss()

        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args

        if dist.is_initialized():
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(self, encoder_inputs, encoder_labels, decoder_inputs, decoder_labels, **kwargs):
        encoder_out: MaskedLMOutput = self.encoder(
            **encoder_inputs,
            labels=encoder_labels,
            output_hidden_states=True,
            return_dict=True
        )

        encoder_loss = encoder_out.loss

        if self.model_args.bottlenecked_pretrain:
            attention_mask = self.decoder.get_extended_attention_mask(
                decoder_inputs['attention_mask'],
                decoder_inputs['attention_mask'].shape,
                decoder_inputs['attention_mask'].device
            )


            encoder_cls_hiddens = encoder_out.hidden_states[-1][:, :1]

            decoder_input_embeds = self.decoder.bert.embeddings(decoder_inputs['input_ids'])
            decoder_input_embeds[:, :1] = encoder_cls_hiddens
            decoder_input_mlm = self.decoder.bert.encoder(decoder_input_embeds, attention_mask=attention_mask)[0]

            prediction_scores = self.decoder.cls(decoder_input_mlm)

            # loss_fct = CrossEntropyLoss()  # -100 index = padding token
            decoder_loss = self.cross_entropy(prediction_scores.view(-1, self.decoder.config.vocab_size), decoder_labels.view(-1))
        else:
            decoder_loss = 0
        # # Contrastive loss
        # encoder_out2: MaskedLMOutput = self.encoder(
        #     **decoder_inputs,
        #     labels=encoder_labels,
        #     output_hidden_states=True,
        #     return_dict=True
        # )
        #
        # encoder_cls_hiddens1 = encoder_cls_hiddens[:, 0]
        # encoder_cls_hiddens2 = encoder_out2.hidden_states[-1][:, 0]
        #
        # encoder_cls_hiddens1 = self._dist_gather_tensor(encoder_cls_hiddens1)
        # encoder_cls_hiddens2 = self._dist_gather_tensor(encoder_cls_hiddens2)
        # scores = torch.matmul(encoder_cls_hiddens1, encoder_cls_hiddens2.transpose(0, 1))
        #
        # target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        # target = target * (encoder_cls_hiddens1.size(0) // encoder_cls_hiddens2.size(0))
        # contrastive_loss = (self.cross_entropy(scores, target) + self.cross_entropy(scores.T, target))/2
        #
        # contrastive_loss = contrastive_loss * self.world_size


        return encoder_loss, decoder_loss, 0


    def mlm_loss(self, hiddens, labels):
        pred_scores = self.lm.cls(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1)
        )
        return masked_lm_loss


    @classmethod
    def from_pretrained(
            cls, config: PretrainedConfig,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            training_args: TrainingArguments,
            *args, **kwargs
    ):
        # hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
        # model = cls(hf_model, model_args, data_args, training_args)
        # path = args[0]
        # if os.path.exists(os.path.join(path, 'model.pt')):
        #     logger.info('loading extra weights from local files')
        #     model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
        #     load_result = model.load_state_dict(model_dict, strict=False)
        #
        # return model

        encoder = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path,
                                                       config=config,
                                                       cache_dir=model_args.cache_dir)
        if model_args.bottlenecked_pretrain:
            decoder_config = copy.deepcopy(config)
            decoder_config.num_hidden_layers = model_args.n_head_layers
            decoder = AutoModelForMaskedLM.from_config(config=decoder_config)
            decoder.cls = encoder.cls
            decoder.bert.embeddings = encoder.bert.embeddings
            decoder.load_state_dict(_get_final_layer(model_args.n_head_layers, encoder.state_dict()))
            logger.info(f'Decoder number of parameters: {count_parameters(decoder)}')
        else:
            decoder = None

        model = cls(encoder, decoder, model_args, data_args, training_args)
        logger.info(f'Encoder number of parameters: {count_parameters(encoder)}')

        return model

    @classmethod
    def from_config(
            cls,
            config: PretrainedConfig,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            training_args: TrainingArguments,
    ):
        encoder = AutoModelForMaskedLM.from_config(config)
        if model_args.bottlenecked_pretrain:
            decoder_config = copy.deepcopy(config)
            decoder_config.num_hidden_layers = model_args.n_head_layers
            decoder = AutoModelForMaskedLM.from_config(decoder_config)
            logger.info(f'Decoder number of parameters: {count_parameters(decoder)}')
        else:
            decoder = None

        model = cls(encoder, decoder, model_args, data_args, training_args)
        logger.info(f'Encoder number of parameters: {count_parameters(encoder)}')

        return model

    def save_pretrained(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('encoder')]
        warnings.warn(f'omiting {len(hf_weight_keys)} transformer weights')
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.training_args], os.path.join(output_dir, 'args.pt'))

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors


class ParallelForPretraining(BoderForPretraining):
    def __init__(
            self,
            encoder,
            decoder,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            training_args: TrainingArguments
    ):
        super(ParallelForPretraining, self).__init__(encoder, decoder, model_args, data_args, training_args)
        self.KL = nn.KLDivLoss(reduction="batchmean")
        self.BCE = nn.BCEWithLogitsLoss()

    def _jenson_shannon_divergence(self, logits1, logits2):
        p = torch.softmax(logits1, dim=1)
        q = torch.softmax(logits2, dim=1)
        m = (p + q) / 2
        return (self.KL(F.log_softmax(logits1, dim=1), m) + self.KL(F.log_softmax(logits2, dim=1), m)) / 2

    def forward(self, encoder_inputs, encoder_labels, decoder_inputs, decoder_labels, **kwargs):

        # randomly shaffle the inputs of encoder and decoder.
        if random.random() < 0.5:
            temp_inputs, temp_labels = encoder_inputs, encoder_labels
            encoder_inputs, encoder_labels = decoder_inputs, decoder_labels
            decoder_inputs, decoder_labels = temp_inputs, temp_labels

        # encoder decoder use each other's token ids as bce labels.
        decoder_bce_labels = encoder_inputs.pop('ids_onehot')
        encoder_bce_labels = decoder_inputs.pop('ids_onehot')

        encoder_out: MaskedLMOutput = self.encoder(
            **encoder_inputs,
            labels=encoder_labels,
            output_hidden_states=True,
            return_dict=True
        )
        mlm_loss = encoder_out.loss

        # Bottlenecked loss
        probs = torch.softmax(torch.max(encoder_out.logits, dim=1)[0], dim=1)
        encoder_cls_reps = torch.matmul(probs, self.encoder.get_output_embeddings().weight.detach())

        decoder_input_embeds = self.decoder.bert.embeddings(decoder_inputs['input_ids'])
        decoder_input_embeds[:, :1] = encoder_cls_reps.unsqueeze(1)
        attention_mask = self.decoder.get_extended_attention_mask(
            decoder_inputs['attention_mask'],
            decoder_inputs['attention_mask'].shape,
            decoder_inputs['attention_mask'].device
        )
        decoder_input_mlm = self.decoder.bert.encoder(decoder_input_embeds, attention_mask=attention_mask)[0]

        prediction_scores = self.decoder.cls(decoder_input_mlm)
        # loss_fct = CrossEntropyLoss()  # -100 index = padding token
        decoder_loss = self.cross_entropy(prediction_scores.view(-1, self.decoder.config.vocab_size),
                                          decoder_labels.view(-1))
        mlm_loss += decoder_loss

        # FLOPS loss
        aggregated_rep, _ = torch.max(
            torch.log(1 + torch.relu(encoder_out.logits)) * encoder_inputs['attention_mask'].unsqueeze(-1), dim=1)
        flops_loss = torch.sum(torch.mean(torch.abs(aggregated_rep), dim=0) ** 2)

        # BCE loss
        encoder_logit_masks = (encoder_inputs['attention_mask'].unsqueeze(-1) == 0).repeat(1, 1, encoder_out.logits.shape[-1])
        encoder_out.logits[encoder_logit_masks] = torch.tensor(-torch.inf, dtype=torch.float16, device=encoder_out.logits.device)
        bce_logits, _ = torch.max(encoder_out.logits, dim=1)
        bce_loss = self.BCE(bce_logits, encoder_bce_labels)

        return mlm_loss, flops_loss, bce_loss


    # def forward(self, encoder_inputs1, encoder_labels1, encoder_inputs2, encoder_labels2):
    #
    #     bce_labels1 = encoder_inputs1.pop('ids_onehot')
    #     bce_labels2 = encoder_inputs2.pop('ids_onehot')
    #     from IPython import embed; embed()
    #
    #
    #     encoder_out1: MaskedLMOutput = self.encoder(
    #         **encoder_inputs1,
    #         labels=encoder_labels1,
    #         output_hidden_states=True,
    #         return_dict=True
    #     )
    #
    #     encoder_out2: MaskedLMOutput = self.encoder(
    #         **encoder_inputs2,
    #         labels=encoder_labels2,
    #         output_hidden_states=True,
    #         return_dict=True
    #     )
    #
    #     mlm_loss = (encoder_out1.loss + encoder_out2.loss) / 2
    #
    #
    #     # aggregated_rep1, _ = torch.max(encoder_out1.logits * encoder_inputs1['attention_mask'].unsqueeze(-1), dim=1)
    #     # aggregated_rep2, _ = torch.max(encoder_out2.logits * encoder_inputs2['attention_mask'].unsqueeze(-1), dim=1)
    #     # aggregated_rep1, _ = torch.max(torch.log(1 + torch.relu(encoder_out1.logits)) * encoder_inputs1['attention_mask'].unsqueeze(-1), dim=1)
    #
    #     logits1 = encoder_out1.logits * encoder_inputs1['attention_mask'].unsqueeze(-1)
    #     logits1 = logits1.masked_fill(logits1 == 0, torch.tensor(-torch.inf, dtype=torch.float16))
    #     logits2 = encoder_out2.logits * encoder_inputs2['attention_mask'].unsqueeze(-1)
    #     logits2 = logits2.masked_fill(logits2 == 0, torch.tensor(-torch.inf, dtype=torch.float16))
    #
    #     aggregated_rep1, _ = torch.max(logits1, dim=1)
    #     aggregated_rep2, _ = torch.max(logits2, dim=1)
    #
    #     flops_loss1 = 0.001 * torch.sum(torch.mean(torch.abs(aggregated_rep1), dim=0) ** 2)
    #     flops_loss2 = 0.001 * torch.sum(torch.mean(torch.abs(aggregated_rep2), dim=0) ** 2)
    #     flop_loss = flops_loss1 + flops_loss2
    #
    #     # expansion_loss = self._jenson_shannon_divergence(aggregated_rep1, aggregated_rep2)
    #
    #     expansion_loss = (self.BCE(aggregated_rep1, bce_labels2) + self.BCE(aggregated_rep2, bce_labels1)) / 2
    #
    #     return mlm_loss, flop_loss, expansion_loss

# class RobertaCondenserForPretraining(CondenserForPretraining):
#     def __init__(
#             self,
#             roberta: RobertaModel,
#             model_args: ModelArguments,
#             data_args: DataTrainingArguments,
#             training_args: TrainingArguments
#     ):
#         super(CondenserForPretraining, self).__init__()
#         self.lm = roberta
#         self.c_head = nn.ModuleList(
#             [RobertaLayer(roberta.config) for _ in range(model_args.n_head_layers)]
#         )
#         self.c_head.apply(self.lm._init_weights)
#         # self.mlm_head = BertOnlyMLMHead(bert.config)
#         self.cross_entropy = nn.CrossEntropyLoss()
#
#         self.model_args = model_args
#         self.training_args = training_args
#         self.data_args = data_args
#
#     def mlm_loss(self, hiddens, labels):
#         pred_scores = self.lm.lm_head(hiddens)
#         masked_lm_loss = self.cross_entropy(
#             pred_scores.view(-1, self.lm.config.vocab_size),
#             labels.view(-1)
#         )
#         return masked_lm_loss

# class CoCondenserForPretraining(CondenserForPretraining):
#     def __init__(
#             self,
#             bert: BertModel,
#             model_args: ModelArguments,
#             data_args: DataTrainingArguments,
#             training_args: CoCondenserPreTrainingArguments
#     ):
#         super(CoCondenserForPretraining, self).__init__(bert, model_args, data_args, training_args)
#
#         effective_bsz = training_args.per_device_train_batch_size * self._world_size() * 2
#         target = torch.arange(effective_bsz, dtype=torch.long).view(-1, 2).flip([1]).flatten().contiguous()
#
#         self.register_buffer(
#             'co_target', target
#         )
#
#     def _gather_tensor(self, t: Tensor):
#         all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
#         dist.all_gather(all_tensors, t)
#         all_tensors[self.training_args.local_rank] = t
#         return all_tensors
#
#     def gather_tensors(self, *tt: Tensor):
#         tt = [torch.cat(self._gather_tensor(t)) for t in tt]
#         return tt
#
#     def forward(self, model_input, labels, grad_cache: Tensor = None, chunk_offset: int = None):
#         attention_mask = self.lm.get_extended_attention_mask(
#             model_input['attention_mask'],
#             model_input['attention_mask'].shape,
#             model_input['attention_mask'].device
#         )
#
#         lm_out: MaskedLMOutput = self.lm(
#             **model_input,
#             labels=labels,
#             output_hidden_states=True,
#             return_dict=True
#         )
#
#         cls_hiddens = lm_out.hidden_states[-1][:, :1]
#         if self.training_args.local_rank > -1 and grad_cache is None:
#             co_cls_hiddens = self.gather_tensors(cls_hiddens.squeeze().contiguous())[0]
#         else:
#             co_cls_hiddens = cls_hiddens.squeeze()
#
#         skip_hiddens = lm_out.hidden_states[self.model_args.skip_from]
#         hiddens = torch.cat([cls_hiddens, skip_hiddens[:, 1:]], dim=1)
#
#         for layer in self.c_head:
#             layer_out = layer(
#                 hiddens,
#                 attention_mask,
#             )
#             hiddens = layer_out[0]
#
#         loss = self.mlm_loss(hiddens, labels)
#         if self.model_args.late_mlm:
#             loss += lm_out.loss
#
#         if grad_cache is None:
#             co_loss = self.compute_contrastive_loss(co_cls_hiddens)
#             return loss + co_loss
#         else:
#             loss = loss * (float(hiddens.size(0)) / self.training_args.per_device_train_batch_size)
#             cached_grads = grad_cache[chunk_offset: chunk_offset + co_cls_hiddens.size(0)]
#             surrogate = torch.dot(cached_grads.flatten(), co_cls_hiddens.flatten())
#             return loss, surrogate
#
#     @staticmethod
#     def _world_size():
#         if dist.is_initialized():
#             return dist.get_world_size()
#         else:
#             return 1
#
#     def compute_contrastive_loss(self, co_cls_hiddens):
#         similarities = torch.matmul(co_cls_hiddens, co_cls_hiddens.transpose(0, 1))
#         similarities.fill_diagonal_(float('-inf'))
#         co_loss = F.cross_entropy(similarities, self.co_target) * self._world_size()
#         return


class Reranker(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataTrainingArguments,
                 train_args: TrainingArguments):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.register_buffer(
            'target_label',
            torch.zeros(self.train_args.per_device_train_batch_size, dtype=torch.long)
        )

        if train_args.local_rank >= 0:
            self.world_size = dist.get_world_size()

    def forward(self, batch):
        ranker_out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True)
        logits = ranker_out.logits

        if self.model_args.temperature is not None:
            assert self.model_args.temperature > 0
            logits = logits / self.model_args.temperature

        if self.train_args.collaborative:
            logits = self.dist_gather_tensor(logits)
            logits = logits.view(
                self.world_size,
                self.train_args.per_device_train_batch_size,
                -1  # chunk
            )
            logits = logits.transpose(0, 1).contiguous()

        if self.training:
            scores = logits.view(
                self.train_args.per_device_train_batch_size,
                self.data_args.train_group_size
            )
            loss = self.cross_entropy(scores, self.target_label)
            # if self.train_args.collaborative or self.train_args.distance_cahce:
                # account for avg in all reduce
                # loss = loss.float() * self.world_size

            return SequenceClassifierOutput(
                loss=loss,
                **ranker_out,
            )
        else:
            return ranker_out

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataTrainingArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        reranker = cls(hf_model, model_args, data_args, train_args)
        return reranker


    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class RerankerDC(Reranker):
    def compute_grad(self, scores: torch.Tensor):
        scores = scores.view(
            self.train_args.per_device_train_batch_size,
            self.data_args.train_group_size
        ).detach().requires_grad_()
        loss = self.cross_entropy(scores, self.target_label)
        loss.backward()

        return loss.detach(), scores.grad

    def forward(self, batch, grad_tensor: torch.Tensor = None):
        ranker_out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True)
        logits = ranker_out.logits

        if self.training:
            if grad_tensor is not None:
                return torch.dot(logits.float().flatten(), grad_tensor.flatten())
            else:
                return logits

        else:
            return ranker_out


class RerankerForInference(nn.Module):
    def __init__(
            self,
            hf_model: Optional[PreTrainedModel] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        super().__init__()
        self.hf_model = hf_model
        self.tokenizer = tokenizer

    def tokenize(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def forward(self, batch):
        return self.hf_model(**batch)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path)
        hf_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path)

        hf_model.eval()
        return cls(hf_model, hf_tokenizer)

    def load_pretrained_model(self, pretrained_model_name_or_path, *model_args, **kwargs):
        self.hf_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

    def load_pretrained_tokenizer(self, pretrained_model_name_or_path, *inputs, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, *inputs, **kwargs
        )

