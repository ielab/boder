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

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping
import datasets
from transformers import BertTokenizer, BertTokenizerFast
import warnings
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForWholeWordMask
from textattack.transformations import WordSwapNeighboringCharacterSwap, \
    WordSwapRandomCharacterDeletion, WordSwapRandomCharacterInsertion, \
    WordSwapRandomCharacterSubstitution, WordSwapQWERTY
from textattack.augmentation import Augmenter
from textattack.transformations import CompositeTransformation
from textattack.constraints.pre_transformation import MinWordLength, StopwordModification
from transformers.data.data_collator import _torch_collate_batch, tolist
from arguments import DataTrainingArguments, RerankerTrainingArguments
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers import DataCollatorWithPadding

STOPWORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                 "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
                 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                 "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                 "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
                 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
                 "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
                 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])


class FixWordSwapQWERTY(WordSwapQWERTY):
    def _get_replacement_words(self, word):
        if len(word) <= 1:
            return []

        candidate_words = []

        start_idx = 1 if self.skip_first_char else 0
        end_idx = len(word) - (1 + self.skip_last_char)

        if start_idx >= end_idx:
            return []

        if self.random_one:
            i = random.randrange(start_idx, end_idx + 1)
            if len(self._get_adjacent(word[i])) == 0:
                candidate_word = (
                        word[:i] + random.choice(list(self._keyboard_adjacency.keys())) + word[i + 1:]
                )
            else:
                candidate_word = (
                        word[:i] + random.choice(self._get_adjacent(word[i])) + word[i + 1:]
                )
            candidate_words.append(candidate_word)
        else:
            for i in range(start_idx, end_idx + 1):
                for swap_key in self._get_adjacent(word[i]):
                    candidate_word = word[:i] + swap_key + word[i + 1:]
                    candidate_words.append(candidate_word)

        return candidate_words


@dataclass
class BoderCollator(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    encoder_mlm_probability: float = 0.3
    decoder_mlm_probability: float = 0.5
    random_mask: bool = False

    def __post_init__(self):
        super(BoderCollator, self).__post_init__()

        from transformers import BertTokenizer, BertTokenizerFast
        from transformers import RobertaTokenizer, RobertaTokenizerFast
        if isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            self.whole_word_cand_indexes = self._whole_word_cand_indexes_bert
        elif isinstance(self.tokenizer, (RobertaTokenizer, RobertaTokenizerFast)):
            self.whole_word_cand_indexes = self._whole_word_cand_indexes_roberta
        else:
            raise NotImplementedError(f'{type(self.tokenizer)} collator not supported yet')

        self.specials = self.tokenizer.all_special_tokens

    def _whole_word_cand_indexes_bert(self, input_tokens: List[str]):
        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token in self.specials:
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        return cand_indexes

    def _whole_word_cand_indexes_roberta(self, input_tokens: List[str]):
        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token in self.specials:
                raise ValueError('We expect only raw input for roberta for current implementation')

            if i == 0:
                cand_indexes.append([0])
            elif not token.startswith('\u0120'):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        return cand_indexes

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        cand_indexes = self._whole_word_cand_indexes_bert(input_tokens)

        random.shuffle(cand_indexes)
        encoder_num_to_predict = min(max_predictions,
                                     max(1, int(round(len(input_tokens) * self.encoder_mlm_probability))))
        decoder_num_to_predict = min(max_predictions,
                                     max(1, int(round(len(input_tokens) * self.decoder_mlm_probability))))

        masked_lms = []
        encoder_masked_lms = []
        decoder_masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= max(encoder_num_to_predict, decoder_num_to_predict):
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > max(encoder_num_to_predict, decoder_num_to_predict):
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue

            encoder_add = True if len(encoder_masked_lms) + len(index_set) <= encoder_num_to_predict else False
            decoder_add = True if len(decoder_masked_lms) + len(index_set) <= decoder_num_to_predict else False

            for index in index_set:
                covered_indexes.add(index)
                if encoder_add and decoder_add:
                    encoder_masked_lms.append(index)
                if decoder_add:
                    decoder_masked_lms.append(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        encoder_mask_labels = [1 if i in encoder_masked_lms else 0 for i in range(len(input_tokens))]
        decoder_mask_labels = [1 if i in decoder_masked_lms else 0 for i in range(len(input_tokens))]
        return encoder_mask_labels, decoder_mask_labels

    def _whole_word_mask_random(self, input_tokens: List[str], max_predictions=512, mlm_probability=0.3):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
                "Please refer to the documentation for more information."
            )

        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def _truncate(self, example: List[int]):
        tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)
        if len(example) <= tgt_len:
            return example
        trunc = len(example) - tgt_len
        trunc_left = random.randint(0, trunc)
        trunc_right = trunc - trunc_left

        truncated = example[trunc_left:]
        if trunc_right > 0:
            truncated = truncated[:-trunc_right]

        if not len(truncated) == tgt_len:
            print(len(example), len(truncated), trunc_left, trunc_right, tgt_len, flush=True)
            raise ValueError
        return truncated

    def _pad(self, seq, val=0):
        assert len(seq) <= self.max_seq_length
        return seq + [val for _ in range(self.max_seq_length - len(seq))]

    def __call__(self, examples, return_tensors=None):
        encoded_examples = []
        encoder_mlm_masks = []
        decoder_mlm_masks = []
        masks = []

        for e in examples:
            e = self._truncate(e)
            tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e]
            tokens = []
            for tid in e:
                tokens.append(self.tokenizer._convert_id_to_token(tid))
            if self.random_mask:
                encoder_mlm_mask = self._whole_word_mask_random(tokens, mlm_probability=self.encoder_mlm_probability)
                decoder_mlm_mask = self._whole_word_mask_random(tokens, mlm_probability=self.decoder_mlm_probability)
            else:
                encoder_mlm_mask, decoder_mlm_mask = self._whole_word_mask(tokens)
            encoder_mlm_mask = self._pad(encoder_mlm_mask)
            encoder_mlm_masks.append(encoder_mlm_mask)
            decoder_mlm_mask = self._pad(decoder_mlm_mask)
            decoder_mlm_masks.append(decoder_mlm_mask)

            encoded = self.tokenizer.encode_plus(
                e,
                add_special_tokens=False,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False
            )

            encoded_examples.append(encoded['input_ids'])
            masks.append(encoded['attention_mask'])

        encoder_inputs, encoder_labels = self.torch_mask_tokens(
            torch.tensor(encoded_examples, dtype=torch.long),
            torch.tensor(encoder_mlm_masks, dtype=torch.long)
        )

        decoder_inputs, decoder_labels = self.torch_mask_tokens(
            torch.tensor(encoded_examples, dtype=torch.long),
            torch.tensor(decoder_mlm_masks, dtype=torch.long)
        )

        encoder_batch = {
            "input_ids": encoder_inputs,
            "labels": encoder_labels,
            "attention_mask": torch.tensor(masks),
        }
        decoder_batch = {
            "input_ids": decoder_inputs,
            "labels": decoder_labels,
            "attention_mask": torch.tensor(masks),
        }

        return encoder_batch, decoder_batch


@dataclass
class TypoBoderCollator(BoderCollator):
    augment_probability: float = 0.1

    def __post_init__(self):
        super(TypoBoderCollator, self).__post_init__()
        transformation = CompositeTransformation([
            WordSwapRandomCharacterDeletion(),
            WordSwapNeighboringCharacterSwap(),
            WordSwapRandomCharacterInsertion(),
            WordSwapRandomCharacterSubstitution(),
            FixWordSwapQWERTY(),
        ])
        constraints = [MinWordLength(3), StopwordModification(STOPWORDS)]
        self.augmenter = Augmenter(transformation=transformation, constraints=constraints, pct_words_to_swap=0)

    def _whole_word_cand_tokens_bert(self, input_tokens: List[str]):
        cand_tokens = []
        for token in input_tokens:
            if token in self.specials:
                continue

            if len(cand_tokens) >= 1 and token.startswith("##"):
                cand_tokens[-1].append(token)
            else:
                cand_tokens.append([token])
        return cand_tokens

    def _whole_word_mask_cands(self, encoder_cand_tokens: List[List[str]], decoder_cand_tokens: List[List[str]],
                               decoder_cand_masks):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        cand_indexes = list(range(len(encoder_cand_tokens)))
        random.shuffle(cand_indexes)

        encoder_length = 0
        decoder_length = 0
        for i, tokens in enumerate(encoder_cand_tokens):
            encoder_length += len(tokens)
            decoder_length += len(decoder_cand_tokens[i])

        encoder_num_to_predict = max(1, int(round(encoder_length * self.encoder_mlm_probability)))
        decoder_num_to_predict = max(1, int(round(decoder_length * self.decoder_mlm_probability)))

        encoder_masked_num = 0
        decoder_masked_num = 0
        encoder_masked_lms = []
        decoder_masked_lms = []

        for ind in cand_indexes:
            if decoder_cand_masks[ind] == 1:
                decoder_masked_lms.append(ind)
                decoder_masked_num += len(decoder_cand_tokens[ind])
                # encoder_masked_num += len(encoder_cand_tokens[ind])

        for ind in cand_indexes:
            if decoder_cand_masks[ind] == 1:
                continue

            if encoder_masked_num >= encoder_num_to_predict and decoder_masked_num >= decoder_num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            # if (decoder_masked_num + len(decoder_cand_tokens[ind])) > \
            #         decoder_num_to_predict or (encoder_masked_num + len(encoder_cand_tokens[ind])) > \
            #         encoder_num_to_predict:
            #     continue

            encoder_add = True if encoder_masked_num + len(
                encoder_cand_tokens[ind]) <= encoder_num_to_predict else False
            decoder_add = True if decoder_masked_num + len(
                decoder_cand_tokens[ind]) <= decoder_num_to_predict else False

            if encoder_add:
                encoder_masked_lms.append(ind)
                encoder_masked_num += len(encoder_cand_tokens[ind])

            if decoder_add:
                decoder_masked_lms.append(ind)
                decoder_masked_num += len(decoder_cand_tokens[ind])

        encoder_mask_labels = []
        decoder_mask_labels = []
        for i in range(len(cand_indexes)):
            if i in encoder_masked_lms:
                encoder_mask_labels.extend([1] * len(encoder_cand_tokens[i]))
            else:
                encoder_mask_labels.extend([0] * len(encoder_cand_tokens[i]))

            if i in decoder_masked_lms:
                decoder_mask_labels.extend([1] * len(decoder_cand_tokens[i]))
            else:
                decoder_mask_labels.extend([0] * len(decoder_cand_tokens[i]))

        return encoder_mask_labels, decoder_mask_labels

    def _pad_n_trunc(self, seq, val=0):
        tgt_len = self.max_seq_length
        if len(seq) <= tgt_len:
            return seq + [val for _ in range(tgt_len - len(seq))]
        else:
            return seq[:tgt_len - 1] + [val]

    def __call__(self, examples, return_tensors=None):
        encoded_encoder_examples = []
        encoded_decoder_examples = []
        encoder_mlm_masks = []
        decoder_mlm_masks = []
        encoder_masks = []
        decoder_masks = []

        for e in examples:
            # e_trunc = self._truncate(e)
            # tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e]
            # encoder_mlm_mask, decoder_mlm_mask = self._whole_word_mask(tokens)
            #
            # encoder_mlm_mask = self._pad([0] + encoder_mlm_mask)
            # encoder_mlm_masks.append(encoder_mlm_mask)
            # decoder_mlm_mask = self._pad([0] + decoder_mlm_mask)
            # decoder_mlm_masks.append(decoder_mlm_mask)

            decoder_tokens = [self.tokenizer._convert_id_to_token(tid) for tid in
                              e[1:-1]]  # do not include [CLS] and [SEP]
            decoder_cand_tokens = self._whole_word_cand_tokens_bert(decoder_tokens)

            decoder_cand_token_mask = []
            # if random.random() < 0.5:
            encoder_cand_tokens = []
            encoder_tokens = []
            for tokens in decoder_cand_tokens:
                if random.random() < self.augment_probability:
                    word = self.tokenizer.convert_tokens_to_string(tokens)
                    aug_word = self.augmenter.augment(word)[0]
                    if aug_word != word:
                        aug_tokens = self.tokenizer.tokenize(aug_word)
                        encoder_cand_tokens.append(aug_tokens)
                        decoder_cand_token_mask.append(1)
                        encoder_tokens.extend(aug_tokens)
                    else:
                        encoder_cand_tokens.append(tokens)
                        decoder_cand_token_mask.append(0)
                        encoder_tokens.extend(tokens)


                else:
                    encoder_cand_tokens.append(tokens)
                    decoder_cand_token_mask.append(0)
                    encoder_tokens.extend(tokens)
            # else:
            #     encoder_cand_tokens = decoder_cand_tokens
            #     encoder_tokens = decoder_tokens
            #     decoder_cand_token_mask = [0 for _ in range(len(encoder_cand_tokens))]

            assert len(decoder_cand_tokens) == len(encoder_cand_tokens)

            if self.random_mask:
                # decoder_tokens = encoder_tokens
                encoder_mlm_mask = self._whole_word_mask_random(encoder_tokens,
                                                                mlm_probability=self.encoder_mlm_probability)
                decoder_mlm_mask = self._whole_word_mask_random(decoder_tokens,
                                                                mlm_probability=self.decoder_mlm_probability)
            else:
                # decoder_tokens = encoder_tokens
                # encoder_mlm_mask, decoder_mlm_mask = self._whole_word_mask(decoder_tokens)

                encoder_mlm_mask, decoder_mlm_mask = self._whole_word_mask_cands(
                    encoder_cand_tokens,
                    decoder_cand_tokens,
                    decoder_cand_token_mask
                )

            encoder_mlm_mask = self._pad_n_trunc([0] + encoder_mlm_mask)
            encoder_mlm_masks.append(encoder_mlm_mask)
            decoder_mlm_mask = self._pad_n_trunc([0] + decoder_mlm_mask)
            decoder_mlm_masks.append(decoder_mlm_mask)

            encoded_encoder = self.tokenizer.encode_plus(
                encoder_tokens,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False
            )

            encoded_decoder = self.tokenizer.encode_plus(
                decoder_tokens,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False
            )

            encoded_encoder_examples.append(encoded_encoder['input_ids'])
            encoder_masks.append(encoded_encoder['attention_mask'])

            encoded_decoder_examples.append(encoded_decoder['input_ids'])
            decoder_masks.append(encoded_decoder['attention_mask'])

        encoder_inputs, encoder_labels = self.torch_mask_tokens(
            torch.tensor(encoded_encoder_examples, dtype=torch.long),
            torch.tensor(encoder_mlm_masks, dtype=torch.long)
        )

        decoder_inputs, decoder_labels = self.torch_mask_tokens(
            torch.tensor(encoded_decoder_examples, dtype=torch.long),
            torch.tensor(decoder_mlm_masks, dtype=torch.long)
        )

        encoder_batch = {
            "input_ids": encoder_inputs,
            "labels": encoder_labels,
            "attention_mask": torch.tensor(encoder_masks),
        }
        decoder_batch = {
            "input_ids": decoder_inputs,
            "labels": decoder_labels,
            "attention_mask": torch.tensor(decoder_masks),
        }

        return encoder_batch, decoder_batch


class TrainDataset(Dataset):
    def __init__(
            self,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
    ):
        self.train_data = dataset
        self.tokenizer = tokenizer
        self.total_len = len(self.train_data) if self.train_data is not None else 0

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        if 'source_input_ids' in self.train_data[item]:
            text = self.train_data[item]['source_input_ids']
        elif 'text' in self.train_data[item]:
            text = self.train_data[item]['text']
            if 'title' in self.train_data[item]:
                text = self.train_data[item]['title'] + ' ' + text
            text = self.tokenizer.encode(self.train_data[item]['text'])
        else:
            raise ValueError('No text found in dataset')
        return text


class ParallelTrainDataset(TrainDataset):
    def __getitem__(self, item):
        tokens1 = self.train_data[item]['tokens1']
        tokens2 = self.train_data[item]['tokens2']
        return tokens1, tokens2


class ParallelCollator(BoderCollator):
    def _truncate(self, example: List[str]):
        tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)
        if len(example) <= tgt_len:
            return example
        truncated = example[:tgt_len]

        return truncated

    def __call__(self, examples, return_tensors=None):
        encoded_examples1 = []
        encoded_examples2 = []
        mlm_masks1 = []
        mlm_masks2 = []
        masks1 = []
        masks2 = []
        bce_labels1 = []
        bce_labels2 = []

        for tokens1, tokens2 in examples:
            tokens1 = self._truncate(tokens1)
            tokens2 = self._truncate(tokens2)

            token_ids1 = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens1), dtype=torch.long)
            bce_labels1.append(torch.zeros(self.tokenizer.vocab_size).scatter_(0, token_ids1, 1.))
            token_ids2 = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens2), dtype=torch.long)
            bce_labels2.append(torch.zeros(self.tokenizer.vocab_size).scatter_(0, token_ids2, 1.))

            mlm_mask1 = self._whole_word_mask_random(tokens1, mlm_probability=self.encoder_mlm_probability)
            mlm_mask2 = self._whole_word_mask_random(tokens2, mlm_probability=self.encoder_mlm_probability)

            mlm_mask1 = self._pad(mlm_mask1)
            mlm_masks1.append(mlm_mask1)
            mlm_mask2 = self._pad(mlm_mask2)
            mlm_masks2.append(mlm_mask2)

            encoded1 = self.tokenizer.encode_plus(
                tokens1,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False
            )
            encoded2 = self.tokenizer.encode_plus(
                tokens2,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False
            )

            encoded_examples1.append(encoded1['input_ids'])
            encoded_examples2.append(encoded2['input_ids'])

            masks1.append(encoded1['attention_mask'])
            masks2.append(encoded2['attention_mask'])

        encoder_inputs1, encoder_labels1 = self.torch_mask_tokens(
            torch.tensor(encoded_examples1, dtype=torch.long),
            torch.tensor(mlm_masks1, dtype=torch.long)
        )

        encoder_inputs2, encoder_labels2 = self.torch_mask_tokens(
            torch.tensor(encoded_examples2, dtype=torch.long),
            torch.tensor(mlm_masks2, dtype=torch.long)
        )

        encoder_batch1 = {
            "input_ids": encoder_inputs1,
            "labels": encoder_labels1,
            "attention_mask": torch.tensor(masks1),
            "ids_onehot": torch.stack(bce_labels1)
        }
        encoder_batch2 = {
            "input_ids": encoder_inputs2,
            "labels": encoder_labels2,
            "attention_mask": torch.tensor(masks2),
            "ids_onehot": torch.stack(bce_labels2)
        }
        return encoder_batch1, encoder_batch2


class GroupedTrainDataset(Dataset):
    query_columns = ['qid', 'query']
    document_columns = ['pid', 'passage']

    def __init__(
            self,
            args: DataTrainingArguments,
            path_to_tsv: Union[List[str], str],
            tokenizer: PreTrainedTokenizer,
            train_args: RerankerTrainingArguments = None,
            cache_dir=None
    ):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_tsv,
            ignore_verifications=False,
            cache_dir=cache_dir
        )['train']

        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)
        self.train_args = train_args

        if train_args is not None and train_args.collaborative:
            import torch.distributed as dist
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            chunk_size = int(self.args.train_group_size / self.world_size)
            self.chunk_start = self.rank * chunk_size
            self.chunk_end = self.chunk_start + chunk_size

    def create_one_example(self, qry_encoding: List[int], doc_encoding: List[int]):
        item = self.tok.encode_plus(
            qry_encoding,
            doc_encoding,
            truncation='only_second',
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> List[BatchEncoding]:
        group = self.nlp_dataset[item]
        examples = []
        group_batch = []
        qry = group['query']
        pos_psg = random.choice(group['positives'])
        examples.append((qry, pos_psg))

        if len(group['negatives']) < self.args.train_group_size - 1:
            negs = random.choices(group['negatives'], k=self.args.train_group_size - 1)
        else:
            negs = random.sample(group['negatives'], k=self.args.train_group_size - 1)

        for neg_entry in negs:
            examples.append((qry, neg_entry))

        # collaborative mode, split the group
        if self.train_args is not None and self.train_args.collaborative:
            examples = examples[self.chunk_start: self.chunk_end]

        for e in examples:
            group_batch.append(self.create_one_example(*e))
        return group_batch


class PredictionDataset(Dataset):
    columns = [
        'qid', 'pid', 'qry', 'psg'
    ]

    def __init__(self, path_to_json: List[str], tokenizer: PreTrainedTokenizer, max_len=128):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_json,
        )['train']
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.nlp_dataset)

    def __getitem__(self, item):
        qid, pid, qry, psg = (self.nlp_dataset[item][f] for f in self.columns)
        return self.tok.encode_plus(
            qry,
            psg,
            truncation='only_second',
            max_length=self.max_len,
            padding=False,
        )


@dataclass
class GroupCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    def __call__(
            self, features
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if isinstance(features[0], list):
            features = sum(features, [])
        return super().__call__(features)
