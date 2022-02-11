# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" GLUE processors and helpers """

import logging
import os

# from ...file_utils import is_tf_available
# from .utils import DataProcessor, InputExample, InputFeatures

from dataclasses import dataclass
from typing import Optional
import json
import copy
import numpy
import torch
from torch.utils.data import Dataset
import sys

import scipy.stats
sigma = 2.5
mu = 0.0

logger = logging.getLogger(__name__)

norm_0 = scipy.stats.norm(mu, sigma).cdf(1)-scipy.stats.norm(mu, sigma).cdf(0)  # norm高斯概率分布函数#目标实体旁边第一个单词的概率
norm_1 = scipy.stats.norm(mu, sigma).cdf(2)-scipy.stats.norm(mu, sigma).cdf(1)  # scipy.stats.norm(mu,sigma)高斯概率密度函数cdf:# Cumulative distribution function.#目标实体旁边第二个单词的概率
norm_2 = scipy.stats.norm(mu, sigma).cdf(3)-scipy.stats.norm(mu, sigma).cdf(2)
norm_3 = scipy.stats.norm(mu, sigma).cdf(4)-scipy.stats.norm(mu, sigma).cdf(3)
norm_4 = scipy.stats.norm(mu, sigma).cdf(5)-scipy.stats.norm(mu, sigma).cdf(4)
norm_5 = scipy.stats.norm(mu, sigma).cdf(6)-scipy.stats.norm(mu, sigma).cdf(5)
norm_6 = scipy.stats.norm(mu, sigma).cdf(7)-scipy.stats.norm(mu, sigma).cdf(6)

# if is_tf_available():
#     import tensorflow as tf

logger = logging.getLogger(__name__)

# tbreak ./transformers/data/processors/cdr.py:206

class CDR_Dataset(Dataset):
    def __init__(self, loader, ex_index2graph, model_type, tokenizer, max_length, max_ent_cnt):
        self.loader = loader
        self.ex_index2graph = ex_index2graph
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_ent_cnt = max_ent_cnt
        self.model_type = model_type
        self.doc_list = [doc_id for doc_id in self.loader.documents.keys()]

        self.ner_map = {'PAD': 0, 'Chemical': 1, 'Disease': 2}
        distance_buckets = numpy.zeros((512), dtype='int64')
        distance_buckets[1] = 1
        distance_buckets[2:] = 2
        distance_buckets[4:] = 3
        distance_buckets[8:] = 4
        distance_buckets[16:] = 5
        distance_buckets[32:] = 6
        distance_buckets[64:] = 7
        distance_buckets[128:] = 8
        distance_buckets[256:] = 9
        self.distance_buckets = distance_buckets

    def __getitem__(self, index):
        doc_id = self.doc_list[index]
        dependency_graph = self.ex_index2graph[doc_id]
        document = self.loader.documents[doc_id]
        entities = self.loader.entities[doc_id]
        labels = self.loader.pairs[doc_id]
        feature = cdr_convert_single_example_to_features(document,
                                                         entities,
                                                         labels,
                                                         dependency_graph,
                                                         self.ner_map,
                                                         self.distance_buckets,
                                                         self.tokenizer,
                                                         self.max_length,
                                                         self.max_ent_cnt,
                                                         self.model_type)

        return feature

    def __len__(self):
        return len(self.loader.documents.keys())

def norm_mask(input_mask):
    output_mask = numpy.zeros(input_mask.shape)
    for i in range(len(input_mask)):
        if not numpy.all(input_mask[i] == 0):
            output_mask[i] = input_mask[i] / sum(input_mask[i])
    return output_mask

def cdr_convert_single_example_to_features(
        document,
        entities,
        labels,
        dependency_graph,
        ner_map,
        distance_buckets,
        tokenizer,
        max_length=512,
        max_ent_cnt=42,
        model_type=None,
        pad_token=0,
):

    input_tokens = []
    tok_to_sent = []
    tok_to_word = []
    word_idx_global = 0
    for sent_idx, sent in enumerate(document):
        for word_idx, word in enumerate(sent):
            tokens_tmp = tokenizer.tokenize(word, add_prefix_space=True)
            input_tokens += tokens_tmp
            tok_to_sent += [sent_idx] * len(tokens_tmp)
            tok_to_word += [word_idx_global] * len(tokens_tmp)
            word_idx_global += 1

    if len(input_tokens) <= max_length - 2:
        if model_type == 'roberta':
            input_tokens = [tokenizer.bos_token] + input_tokens + [tokenizer.eos_token]
        else:
            input_tokens = [tokenizer.cls_token] + input_tokens + [tokenizer.sep_token]
        tok_to_sent = [None] + tok_to_sent + [None]
        tok_to_word = [None] + tok_to_word + [None]
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        # padding
        padding = [None] * (max_length - len(input_ids))
        tok_to_sent += padding
        tok_to_word += padding
        padding = [0] * (max_length - len(input_ids))
        attention_mask += padding
        token_type_ids += padding
        padding = [pad_token] * (max_length - len(input_ids))
        input_ids += padding
    else:
        input_tokens = input_tokens[:max_length - 2]
        tok_to_sent = tok_to_sent[:max_length - 2]
        tok_to_word = tok_to_word[:max_length - 2]
        if model_type == 'roberta':
            input_tokens = [tokenizer.bos_token] + input_tokens + [tokenizer.eos_token]
        else:
            input_tokens = [tokenizer.cls_token] + input_tokens + [tokenizer.sep_token]
        tok_to_sent = [None] + tok_to_sent + [None]
        tok_to_word = [None] + tok_to_word + [None]
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [pad_token] * len(input_ids)

    # ent_mask & ner / coreference feature
    ent_mask = numpy.zeros((max_ent_cnt, max_length), dtype='float')
    ent_ner = [0] * max_length
    ent_pos = [0] * max_length
    tok_to_ent = [-1] * max_length
    for ent_idx, ent in enumerate(entities.keys()):
        mention_start = entities[ent].mstart.split(':')
        mention_end = entities[ent].mend.split(':')
        mention_sent = entities[ent].sentNo.split(':')
        for mention_idx in range(len(mention_start)):
            for tok_idx in range(len(input_ids)):
                if tok_to_sent[tok_idx] == int(mention_sent[mention_idx]) and int(mention_start[mention_idx]) <= tok_to_word[tok_idx] < int(mention_end[mention_idx]):
                    ent_mask[ent_idx][tok_idx] = 1
                    ent_ner[tok_idx] = ner_map[entities[ent].type]
                    ent_pos[tok_idx] = ent_idx + 1
                    tok_to_ent[tok_idx] = ent_idx

    # Gauss
    gauss_p = numpy.zeros(512)
    for entity_idx, entity in enumerate(entities.keys()):
        for token_idx in range(len(input_ids)):
            if ent_mask[entity_idx][token_idx] == 1:
                if token_idx < 512:
                    gauss_p[token_idx] = norm_0
                if token_idx + 1 < 512:
                    gauss_p[token_idx + 1] = norm_0
                if token_idx + 2 < 512:
                    gauss_p[token_idx + 2] = norm_1
                if token_idx + 3 < 512:
                    gauss_p[token_idx + 3] = norm_2
                if token_idx + 4 < 512:
                    gauss_p[token_idx + 4] = norm_3
                if token_idx + 5 < 512:
                    gauss_p[token_idx + 5] = norm_4
                if token_idx + 6 < 512:
                    gauss_p[token_idx + 6] = norm_5
                if token_idx + 7 < 512:
                    gauss_p[token_idx + 7] = norm_6
                if token_idx - 1 > 0:
                    gauss_p[token_idx - 1] = norm_1
                if token_idx - 2 > 0:
                    gauss_p[token_idx - 2] = norm_2
                if token_idx - 3 > 0:
                    gauss_p[token_idx - 3] = norm_3
                if token_idx - 4 > 0 and gauss_p[token_idx - 4] == 0:
                    gauss_p[token_idx - 4] = norm_4
                if token_idx - 5 > 0 and gauss_p[token_idx - 5] == 0:
                    gauss_p[token_idx - 5] = norm_5
                if token_idx - 6 > 0 and gauss_p[token_idx - 6] == 0:
                    gauss_p[token_idx - 6] = norm_6

    # distance feature
    ent_first_appearance = [0] * max_ent_cnt
    ent_distance = numpy.zeros((max_ent_cnt, max_ent_cnt), dtype='int8')  # padding id is 10
    for i in range(len(entities)):
        if numpy.all(ent_mask[i] == 0):
            continue
        else:
            ent_first_appearance[i] = numpy.where(ent_mask[i] == 1)[0][0]
    for i in range(len(entities)):
        for j in range(len(entities)):
            if ent_first_appearance[i] != 0 and ent_first_appearance[j] != 0:
                if ent_first_appearance[i] >= ent_first_appearance[j]:
                    ent_distance[i][j] = distance_buckets[ent_first_appearance[i] - ent_first_appearance[j]]
                else:
                    ent_distance[i][j] = - distance_buckets[- ent_first_appearance[i] + ent_first_appearance[j]]
    ent_distance += 10  # norm from [-9, 9] to [1, 19]

    # structure prior for attentive biase
    # PRIOR DEFINITION  | share ent context | diff ent context | No ent
    # share sem context |         1         |        2         |   3
    # diff sem context  |         4         |        5 (1~4)   |   6
    structure_mask = numpy.zeros((5, max_length, max_length), dtype='float')
    for i in range(max_length):
        if attention_mask[i] == 0:
            break
        else:
            if tok_to_ent[i] != -1:
                for j in range(max_length):
                    if tok_to_sent[j] is None:
                        continue
                    # intra
                    if tok_to_sent[j] == tok_to_sent[i]:
                        # intra-coref
                        if tok_to_ent[j] == tok_to_ent[i]:
                            structure_mask[0][i][j] = 1
                        # intra-relate
                        elif tok_to_ent[j] != -1:
                            structure_mask[1][i][j] = 1
                        # intra-NA
                        else:
                            structure_mask[2][i][j] = 1
                    # inter
                    else:
                        # inter-coref
                        if tok_to_ent[j] == tok_to_ent[i]:
                            structure_mask[3][i][j] = 1
                        # inter-relate
                        elif tok_to_ent[j] != -1:
                            structure_mask[4][i][j] = 1

    # label
    # label_ids = numpy.full((max_ent_cnt, max_ent_cnt), -1, dtype='int64')
    label_map = {'1:CID:2': 1, '1:NR:2': 0}
    label_ids = numpy.zeros((max_ent_cnt, max_ent_cnt, 2), dtype='bool')
    # label_mask = numpy.zeros((max_ent_cnt, max_ent_cnt), dtype='bool')
    entities_list = [ent_id for ent_id in entities.keys()]
    chemical_ent_count = sum([1 for ent in entities_list if entities[ent].type == 'Chemical'])
    disease_ent_count = sum([1 for ent in entities_list if entities[ent].type == 'Disease'])
    assert len(labels) == chemical_ent_count * disease_ent_count
    for ent_pair in labels.keys():
        s_id = entities_list.index(ent_pair[0])
        o_id = entities_list.index(ent_pair[1])
        if labels[ent_pair].direction == 'L2R':
            ent_h = s_id
            ent_t = o_id
        elif labels[ent_pair].direction == 'R2L':
            ent_h = o_id
            ent_t = s_id
        # ent_h = s_id
        # ent_t = o_id
        if labels[ent_pair].type == '1:NR:2':
            label_ids[ent_h][ent_t][label_map['1:NR:2']] = 1
            # label_mask[ent_h][ent_t] = 1
        elif labels[ent_pair].type == '1:CID:2':
            label_ids[ent_h][ent_t][label_map['1:CID:2']] = 1
            # label_mask[ent_h][ent_t] = 1
        elif labels[ent_pair].type == 'not_include':
            continue
        else:
            sys.exit('unexpected relation label!')

    label_mask = numpy.zeros((max_ent_cnt, max_ent_cnt), dtype='bool')
    label_mask[:len(entities), :len(entities)] = 1
    for ent in range(len(entities)):
        label_mask[ent][ent] = 0
        if numpy.all(ent_mask[ent] == 0):
            label_mask[ent, :] = 0
            label_mask[:, ent] = 0

    ent_mask = norm_mask(ent_mask)

    # built dependency graph
    dependency_graph0 = dependency_graph[0]
    dependency_graph1 = dependency_graph[1]
    adj0 = numpy.pad(dependency_graph0,
                     ((1, (max_length - 1) - (len(dependency_graph0))),
                      (1, (max_length - 1) - (len(dependency_graph0)))),
                     'constant')
    adj1 = numpy.pad(dependency_graph1,
                     ((1, (max_length - 1) - (len(dependency_graph1))),
                      (1, (max_length - 1) - (len(dependency_graph1)))),
                     'constant')
    # adj = adj0 + adj1
    # adj[adj >= 1] = 1
    adj = adj0

    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length
    assert len(token_type_ids) == max_length
    assert ent_mask.shape == (max_ent_cnt, max_length)
    assert label_ids.shape == (max_ent_cnt, max_ent_cnt, 2)
    assert label_mask.shape == (max_ent_cnt, max_ent_cnt)
    assert len(ent_ner) == max_length
    assert len(ent_pos) == max_length
    assert ent_distance.shape == (max_ent_cnt, max_ent_cnt)
    assert structure_mask.shape == (5, max_length, max_length)
    assert adj.shape == (max_length, max_length)
    assert len(gauss_p) == max_length

    # tbreak ./transformers/data/processors/cdr.py:206

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    ent_mask = torch.tensor(ent_mask, dtype=torch.float)
    ent_ner = torch.tensor(ent_ner, dtype=torch.long)
    ent_pos = torch.tensor(ent_pos, dtype=torch.long)
    ent_distance = torch.tensor(ent_distance, dtype=torch.long)
    structure_mask = torch.tensor(structure_mask, dtype=torch.long)
    label_ids = torch.tensor(label_ids, dtype=torch.bool)
    label_mask = torch.tensor(label_mask, dtype=torch.bool)
    adj = torch.tensor(adj, dtype=torch.float)
    gauss_p = torch.tensor(gauss_p, dtype=torch.float)

    # import sys
    # sys.getsizeof(features.input_ids)

    return (input_ids, attention_mask, token_type_ids, ent_mask, ent_ner, ent_pos,
            ent_distance, structure_mask, label_ids, label_mask, adj, gauss_p)