import itertools
import os
from abc import abstractmethod
from argparse import ArgumentParser
from typing import Dict, List

import numpy as np
import pickle5 as pickle
import pytorch_lightning as pl
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from constants import ADJ_KEYS, CHOICE_KEYS, NUM_CHOICES, NUM_RELS
from mmap_dataset import MMapIndexedDataset
from utils import get_logger, load_qa_predictions

logger = get_logger(__name__)


class SalKGDataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.args = params
        self.dataset: Dict[str, DataLoader] = {}

        if self.args.dataset in ["csqa", "qasc"]:
            # OR 'official' if you want to submit to leaderboard
            split_type = "inhouse"
        elif self.args.dataset == "codah":
            split_type = "fold_0"
        else:
            # obqa
            split_type = "official"
        bin_suffix = '' if self.args.train_percentage == 100 else str(self.args.train_percentage)
        self.data_path = os.path.join(self.args.data, self.args.dataset,
                                      split_type, self.args.arch.replace('_', '-'), 'bin' + bin_suffix)
        logger.info(f"data path = {self.data_path}")

    def graph_encoder_keys(self, is_fine_occl=False) -> List[str]:
        """indexed dataset's key for training LM w/ KG"""
        keys = []
        if self.args.graph_encoder in ['rn', 'pathgen']:
            keys += ['qa', 'rel', 'num_tuples']
        elif self.args.graph_encoder == 'mhgrn':
            keys += ['concept', 'node_type', 'adj_len']
            if not is_fine_occl:
                keys += ADJ_KEYS[:NUM_CHOICES[self.args.dataset]]
        return keys

    def fine_occl_keys(self) -> List[str]:
        if self.args.graph_encoder in ['rn', 'pathgen']:
            return ['rn_text', 'rn_label', 'rn_id']
        elif self.args.graph_encoder == 'mhgrn':
            return ['mhgrn_text', 'mhgrn_label', 'mhgrn_id', 'adj']

    @abstractmethod
    def load_dataset(self, split):
        pass

    def setup(self):
        for split in ['train', 'valid', 'test']:
            self.load_dataset(split)

    def train_dataloader(self, shuffle=True):
        return DataLoader(self.dataset['train'], batch_size=self.args.train_batch_size,
                          num_workers=self.args.num_workers,
                          collate_fn=self.dataset['train'].collater, shuffle=shuffle, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset['valid'], batch_size=self.args.eval_batch_size,
                          num_workers=self.args.num_workers,
                          collate_fn=self.dataset['valid'].collater, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers,
                          collate_fn=self.dataset['test'].collater, pin_memory=True)

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--train_batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)
        parser.add_argument("--num_workers", default=10, type=int)
        parser.add_argument("--task", default='qa', type=str, choices=['qa', 'saliency'])
        parser.add_argument("--data", default='./data/', type=str)
        parser.add_argument("--ent_emb_path", default='./data/mhgrn_data/cpnet/tzw.ent.npy', type=str)
        parser.add_argument("--rel_emb_path", default='./data/mhgrn_data/transe/glove.transe.sgd.rel.npy', type=str)
        parser.add_argument("--dataset", default='csqa', type=str, choices=['csqa', 'obqa', 'codah'])
        parser.add_argument("--split_type", default='inhouse', type=str, choices=['inhouse', 'official', 'fold_0'])
        parser.add_argument('--saliency_mode', default='none', type=str, choices=['coarse', 'fine', 'none', 'hybrid'])
        parser.add_argument('--saliency_heuristic', default='none', type=str,
                            choices=['gst', 'ratio', 'topk', 'random', 'qa', 'random_all', 'pos_only_binary', 'raw'])
        parser.add_argument('--saliency_method', default='none', type=str,
                            choices=['grad', 'occl', 'random', 'none', 'goccl', 'qa', 'ig'])
        parser.add_argument('--saliency_source', default='none', type=str, choices=['target', 'pred', 'none'])
        parser.add_argument('--save_saliency', action='store_true')
        parser.add_argument('--saliency_exp', default=None, type=int)
        parser.add_argument('--no_kg_exp', default=None, type=int)
        parser.add_argument('--kg_exp', default=None, type=int)
        parser.add_argument('--train_percentage', default=100, type=int)
        parser.add_argument('--pos_weight', default=-1, type=float)
        parser.add_argument('--coarse_pos_weight', default=-1, type=float)
        parser.add_argument('--fine_pos_weight', default=-1, type=float)
        parser.add_argument('--saliency_value', default=0.0, type=float)
        parser.add_argument('--qa_no_kg_dir', type=str)
        parser.add_argument('--qa_kg_dir', type=str)
        return parser


class QADataModule(SalKGDataModule):
    def load_dataset(self, split):
        # if fine occl
        fine_occl = self.args.save_saliency and self.args.saliency_mode == 'fine' and self.args.saliency_method == 'occl'
        if fine_occl:
            keys = self.fine_occl_keys() + self.graph_encoder_keys(is_fine_occl=True)
            output_suffix = ".fine_occl"
        else:
            keys = ['text', 'label'] + self.graph_encoder_keys(is_fine_occl=False)
            output_suffix = ""

        path_embedding = None
        if self.args.graph_encoder == 'pathgen':
            path_embedding_path = os.path.join(self.data_path,
                                               'path_embedding_{}{}.pickle'.format(split, output_suffix))
            with open(path_embedding_path, 'rb') as handle:
                # (sth, 5, 5, 768)
                path_embedding = pickle.load(handle)

        indexed_datasets = {}
        for key in keys:
            indexed_datasets[key] = MMapIndexedDataset(
                os.path.join(self.data_path, f'{split}.{key}{output_suffix}')
            )

        self.dataset[split] = QADataset(
            dataset=self.args.dataset,
            indexed_datasets=indexed_datasets,
            graph_encoder=self.args.graph_encoder,
            sal_fine_occl=fine_occl,
            path_emb=path_embedding,
        )


class QADataset(Dataset):
    def __init__(
            self,
            dataset, indexed_datasets,
            graph_encoder=None, sal_fine_occl=False, path_emb=None):
        self.dataset = dataset
        self.indexed_datasets = indexed_datasets
        self.graph_encoder = graph_encoder
        self.epoch = 0
        self.num_choices = NUM_CHOICES[dataset]
        self.choice_keys = CHOICE_KEYS[:self.num_choices]
        self.num_rels = NUM_RELS
        self.path_emb = path_emb
        self.sal_fine_occl = sal_fine_occl

        if graph_encoder == 'rn':
            self.max_tuple_num = len(indexed_datasets['rel'][0]) if self.sal_fine_occl else int(
                len(indexed_datasets['rel'][0]) / self.num_choices)
        elif graph_encoder == 'mhgrn':
            self.max_node_num = len(indexed_datasets['concept'][0]) if self.sal_fine_occl else int(
                len(indexed_datasets['concept'][0]) / self.num_choices)
            self.adj_empty = torch.zeros((self.num_choices, self.num_rels - 1, self.max_node_num, self.max_node_num))
            self.adj_empty[:, -1] = torch.eye(self.max_node_num)
        elif graph_encoder == 'pathgen':
            self.max_tuple_num = len(indexed_datasets['rel'][0]) if self.sal_fine_occl else int(
                len(indexed_datasets['rel'][0]) / self.num_choices)

    def __len__(self):
        return len([x for x in self.indexed_datasets.values()][0])

    def __getitem__(self, index):
        item = {}
        item['text'] = self.get_text(index)
        item['target'] = self.get_target(index)
        item['index'] = index

        if self.sal_fine_occl:
            if self.graph_encoder in ['rn', 'pathgen']:
                item['fine_occl_id'] = self.indexed_datasets['rn_id'][index]
            elif self.graph_encoder == 'mhgrn':
                item['fine_occl_id'] = self.indexed_datasets['mhgrn_id'][index]

        if self.graph_encoder == 'rn':
            item = self.get_rn_data(item, index)
        elif self.graph_encoder == 'mhgrn':
            item = self.get_mhgrn_data(item, index)
        elif self.graph_encoder == 'pathgen':
            item = self.get_rn_data(item, index)
            item = self.get_pathgen_data(item, index)

        return item

    def get_text(self, index):
        if self.sal_fine_occl:
            if self.graph_encoder in ['rn', 'pathgen']:
                return self.indexed_datasets['rn_text'][index]
            elif self.graph_encoder == 'mhgrn':
                return self.indexed_datasets['mhgrn_text'][index]
        return self.indexed_datasets['text'][index].reshape(self.num_choices, -1)

    def get_target(self, index):
        if self.sal_fine_occl:
            if self.graph_encoder in ['rn', 'pathgen']:
                return self.indexed_datasets['rn_label'][index]
            elif self.graph_encoder == 'mhgrn':
                return self.indexed_datasets['mhgrn_label'][index]
        return self.indexed_datasets['label'][index]

    def get_rn_data(self, item, index):
        if self.sal_fine_occl:
            qa_ids = self.indexed_datasets['qa'][index].reshape(self.max_tuple_num, 2)
            rel_ids = self.indexed_datasets['rel'][index]
        else:
            qa_ids = self.indexed_datasets['qa'][index].reshape(self.num_choices, self.max_tuple_num, 2)
            rel_ids = self.indexed_datasets['rel'][index].reshape(self.num_choices, self.max_tuple_num)
        num_tuples = self.indexed_datasets['num_tuples'][index]

        item['qa_ids'] = qa_ids
        item['rel_ids'] = rel_ids
        item['num_tuples'] = num_tuples

        return item

    def get_mhgrn_data(self, item, index):
        adj_len = self.indexed_datasets['adj_len'][index]
        if self.sal_fine_occl:
            concept_ids = self.indexed_datasets['concept'][index]
            node_type_ids = self.indexed_datasets['node_type'][index]
            adj_tensor = self.adj_empty[0].clone()
            adj = self.indexed_datasets['adj'][index].reshape(3, -1)
            adj_tensor[adj[0], adj[1], adj[2]] = 1.
        else:
            concept_ids = self.indexed_datasets['concept'][index].reshape(self.num_choices, self.max_node_num)
            node_type_ids = self.indexed_datasets['node_type'][index].reshape(self.num_choices, self.max_node_num)
            adj_tensor = self.adj_empty.clone()
            # indexth adj matrix, adj_A = i, j, k tensor of same length (variable to index)
            # Adj (#relation, #node, #node) where Adj=1
            adj = [
                self.indexed_datasets['adj_{}'.format(key)][index].reshape(3, -1) for key in self.choice_keys
            ]
            # restore Adj as non-sparse matrix
            for choice_id, (rel, concept1, concept2) in enumerate(adj):
                adj_tensor[choice_id, rel, concept1, concept2] = 1.

        item['concept_ids'] = concept_ids
        item['node_type_ids'] = node_type_ids
        item['adj_len'] = adj_len
        item['adj'] = adj_tensor
        return item

    def get_pathgen_data(self, item, index):
        item['path_emb'] = self.path_emb[index]
        return item

    def collater(self, instances):
        batch_size = len(instances)
        if batch_size == 0:
            return None

        texts = [instance['text'] for instance in instances]
        targets = [instance['target'] for instance in instances]
        indices = [instance['index'] for instance in instances]
        batch = {
            'text': torch.stack(texts) if self.sal_fine_occl else torch.cat(texts),
            'target': torch.cat(targets).long(),
            'index': torch.LongTensor(indices),
            'size': batch_size,
        }

        if self.graph_encoder == 'rn':
            batch['qa_ids'] = torch.stack([instance['qa_ids'] for instance in instances]).reshape(-1,
                                                                                                  self.max_tuple_num, 2)
            batch['rel_ids'] = torch.stack([instance['rel_ids'] for instance in instances]).reshape(-1,
                                                                                                    self.max_tuple_num)
            batch['num_tuples'] = torch.stack([instance['num_tuples'] for instance in instances]).reshape(-1)

        elif self.graph_encoder == 'mhgrn':
            batch['concept_ids'] = torch.stack([instance['concept_ids'] for instance in instances]).reshape(-1,
                                                                                                            self.max_node_num)
            batch['node_type_ids'] = torch.stack([instance['node_type_ids'] for instance in instances]).reshape(-1,
                                                                                                                self.max_node_num)
            batch['adj_len'] = torch.stack([instance['adj_len'] for instance in instances]).reshape(-1)
            batch['adj'] = torch.stack([instance['adj'] for instance in instances]).reshape(-1, self.num_rels - 1,
                                                                                            self.max_node_num,
                                                                                            self.max_node_num)

        elif self.graph_encoder == 'pathgen':
            batch['qa_ids'] = torch.stack([instance['qa_ids'] for instance in instances]).reshape(-1,
                                                                                                  self.max_tuple_num, 2)
            batch['rel_ids'] = torch.stack([instance['rel_ids'] for instance in instances]).reshape(-1,
                                                                                                    self.max_tuple_num)
            batch['num_tuples'] = torch.stack([instance['num_tuples'] for instance in instances]).reshape(-1)
            batch['path_emb'] = torch.stack([instance['path_emb'] for instance in instances])

        if self.sal_fine_occl:
            batch['fine_occl_id'] = torch.stack([instance['fine_occl_id'] for instance in instances])

        return batch


class SaliencyDataModule(SalKGDataModule):
    """
    when task=saliency
    """

    def coarse_sal_keys(self):
        """use coarse saliency built from `nokg` and `kg` exp's csv"""
        assert (self.args.no_kg_exp is not None) and (self.args.kg_exp is not None)
        sal_keys = [
            '{}.sal_coarse_{}_{}_cls{}_{}_nokg{}_kg{}_t{}'.format(
                self.args.graph_encoder,
                self.args.saliency_method,
                self.args.saliency_source if not self.args.save_saliency else 'target',
                self.args.sal_num_classes,
                key,
                self.args.no_kg_exp,
                self.args.kg_exp,
                self.args.threshold
            ) for key in CHOICE_KEYS[:NUM_CHOICES[self.args.dataset]]
        ]
        return sal_keys, dict(zip(CHOICE_KEYS[:NUM_CHOICES[self.args.dataset]], sal_keys))

    def fine_sal_keys(self):
        """use fine saliency built from `saliency_exp`'s csv"""
        assert self.args.saliency_exp is not None
        sal_keys = ['{}.sal_fine_{}_{}_{}_{}_{}_FS{}'.format(
            self.args.graph_encoder,
            self.args.saliency_method,
            self.args.saliency_source if not self.args.save_saliency else 'target',
            self.args.saliency_heuristic,
            self.args.saliency_value,
            key,
            self.args.saliency_exp
        ) for key in CHOICE_KEYS[:NUM_CHOICES[self.args.dataset]]]
        return sal_keys, dict(zip(CHOICE_KEYS[:NUM_CHOICES[self.args.dataset]], sal_keys))

    def load_dataset(self, split):
        fine_occl = self.args.save_saliency and self.args.saliency_mode == 'fine' and self.args.saliency_method == 'occl'
        if fine_occl:
            keys = self.fine_occl_keys() + self.graph_encoder_keys(is_fine_occl=True)
            output_suffix = ".fine_occl"
        else:
            keys = ['text', 'label'] + self.graph_encoder_keys(is_fine_occl=False)
            output_suffix = ""

        path_embedding = None
        if self.args.graph_encoder == 'pathgen':
            path_embedding_path = os.path.join(self.data_path,
                                               'path_embedding_{}{}.pickle'.format(split, output_suffix))
            with open(path_embedding_path, 'rb') as handle:
                # (sth, 5, 5, 768)
                path_embedding = pickle.load(handle)

        dataset_keys = {}

        if self.args.saliency_mode == 'coarse':
            sal_keys, sal_results = self.coarse_sal_keys()
            dataset_keys['saliency_results'] = sal_results
            keys += sal_keys

        elif (
                self.args.saliency_mode == 'fine'
                and self.args.saliency_method != 'random'
        ):
            sal_keys, sal_results = self.coarse_sal_keys()
            dataset_keys['saliency_results'] = sal_results
            keys += sal_keys

        elif (
                self.args.saliency_mode == 'hybrid'
                and self.args.saliency_method != 'random'
        ):
            coarse_sal_keys, coarse_sal_results = self.coarse_sal_keys()
            dataset_keys['coarse_saliency_results'] = coarse_sal_results
            keys += coarse_sal_keys

            fine_sal_keys, fine_sal_results = self.fine_sal_keys()
            dataset_keys['fine_saliency_results'] = fine_sal_results
            keys += fine_sal_keys

        qa_prob = None
        if self.args.saliency_mode == 'coarse' and self.args.coarse_model == 'ensemble':
            no_kg_path = os.path.join(self.args.qa_no_kg_dir, f'sal_coarse_occl_target_{split}.csv')
            kg_path = os.path.join(self.args.qa_kg_dir, f'sal_coarse_occl_target_{split}.csv')
            no_kg_probs = load_qa_predictions(no_kg_path)[:, 1:]
            kg_probs = load_qa_predictions(kg_path)[:, 1:]
            qa_prob = np.stack((no_kg_probs, kg_probs), axis=-1)

        elif self.args.saliency_mode == 'hybrid':
            no_kg_path = os.path.join(self.args.qa_no_kg_dir, 'sal_coarse_occl_target_{}.csv'.format(split))
            qa_prob = load_qa_predictions(no_kg_path)[:, 1:]

        indexed_datasets = {}
        for key in keys:
            indexed_datasets[key] = MMapIndexedDataset(
                os.path.join(self.data_path, f'{split}.{key}{output_suffix}')
            )

        self.dataset[split] = SaliencyDataset(
            dataset=self.args.dataset,
            indexed_datasets=indexed_datasets,
            dataset_keys=dataset_keys,
            qa_prob=qa_prob,
            criterion=self.args.criterion,
            graph_encoder=self.args.graph_encoder,
            sal_gnn_layer_type=self.args.sal_gnn_layer_type,
            saliency_mode=self.args.saliency_mode,
            sal_fine_occl=fine_occl,
            saliency_heuristic=self.args.saliency_heuristic,
            sal_num_classes=self.args.sal_num_classes,
            pos_weight=self.args.pos_weight,
            coarse_pos_weight=self.args.coarse_pos_weight,
            fine_pos_weight=self.args.fine_pos_weight,
            path_embs=path_embedding,
        )


class SaliencyDataset(Dataset):
    def __init__(
            self, dataset, indexed_datasets, dataset_keys,
            qa_prob, criterion,
            graph_encoder=None, sal_gnn_layer_type=None,
            sal_fine_occl=False, saliency_mode=None, saliency_heuristic=None, sal_num_classes=None,
            pos_weight=1, coarse_pos_weight=1, fine_pos_weight=1,
            path_embs=None,
    ):
        self.dataset = dataset
        self.criterion = criterion
        self.indexed_datasets = indexed_datasets
        self.dataset_keys = dataset_keys
        self.qa_prob = qa_prob
        self.graph_encoder = graph_encoder
        self.epoch = 0
        self.num_choices = NUM_CHOICES[dataset]
        self.choice_keys = CHOICE_KEYS[:self.num_choices]
        self.num_rels = NUM_RELS
        self.saliency_mode = saliency_mode
        self.saliency_heuristic = saliency_heuristic
        self.sal_num_classes = sal_num_classes
        self.pos_weight = pos_weight
        self.coarse_pos_weight = coarse_pos_weight
        self.fine_pos_weight = fine_pos_weight
        self.sal_gnn_layer_type = sal_gnn_layer_type
        self.path_emb = path_embs

        if saliency_mode == 'hybrid':
            self.coarse_pos_weight = self.get_pos_weight(self.coarse_pos_weight, 'coarse')
            self.fine_pos_weight = self.get_pos_weight(self.fine_pos_weight, 'fine')
        elif saliency_heuristic == "raw":
            # not usinig pos weight
            self.pos_weight = -1
        else:
            self.pos_weight = self.get_pos_weight(self.pos_weight, saliency_mode)

        self.sal_fine_occl = sal_fine_occl

        if graph_encoder == 'rn':
            self.max_tuple_num = int(len(indexed_datasets['rel'][0]) / self.num_choices)
        elif graph_encoder == 'mhgrn':
            self.max_node_num = int(len(indexed_datasets['concept'][0]) / self.num_choices)
            self.adj_empty = torch.zeros((self.num_choices, self.num_rels - 1, self.max_node_num, self.max_node_num))
            self.adj_empty[:, -1] = torch.eye(self.max_node_num)
        elif graph_encoder == 'pathgen':
            self.max_tuple_num = int(len(indexed_datasets['rel'][0]) / self.num_choices)

    def __len__(self):
        return len([x for x in self.indexed_datasets.values()][0])

    def get_pos_weight(self, pos_weight, saliency_mode):
        assert saliency_mode in ['coarse', 'fine']
        if pos_weight > 0:
            return pos_weight
        elif self.target_type == 'cls' and self.sal_num_classes == 2:
            saliency_targets = []
            for index in range(len(self)):
                for key in self.choice_keys:
                    if saliency_mode == 'coarse':
                        if 'saliency_results' in self.dataset_keys:
                            saliency_targets.append(
                                self.indexed_datasets[self.dataset_keys['saliency_results'][key]][index]
                            )
                        else:
                            saliency_targets.append(
                                self.indexed_datasets[self.dataset_keys['coarse_saliency_results'][key]][index]
                            )
                    else:
                        if self.saliency_heuristic in ['gst', 'topk', 'ratio', 'random', 'qa', 'pos_only_binary']:
                            if 'saliency_results' in self.dataset_keys:
                                saliency_targets.append(
                                    self.indexed_datasets[self.dataset_keys['saliency_results'][key]][index]
                                )
                            else:
                                saliency_targets.append(
                                    self.indexed_datasets[self.dataset_keys['fine_saliency_results'][key]][index]
                                )
                        else:
                            raise NotImplementedError
            saliency_targets = np.concatenate(saliency_targets)
            loss_weights = torch.from_numpy(
                compute_class_weight('balanced', classes=np.array([0, 1]), y=saliency_targets)).float()
            return loss_weights[1] / loss_weights[0]
        else:
            return None

    def __getitem__(self, index):
        item = {}
        item['text'] = self.get_text(index)
        item['index'] = index

        if self.saliency_mode == 'hybrid':
            if self.criterion == 'KL_loss':
                item['coarse_sal_target'], item['fine_sal_target'], item['fine_sal_target_flat'] = self.get_target(
                    index)
            else:
                item['coarse_sal_target'], item['fine_sal_target'] = self.get_target(index)
        else:
            if self.criterion == 'KL_loss':
                item['target'], item['target_flat'] = self.get_target(index)
            else:
                item['target'] = self.get_target(index)

        if self.graph_encoder == 'rn':
            item = self.get_rn_data(item, index)
        elif self.graph_encoder == 'mhgrn':
            item = self.get_mhgrn_data(item, index)
        elif self.graph_encoder == 'pathgen':
            item = self.get_rn_data(item, index)
            item = self.get_pathgen_data(item, index)

        if not self.qa_prob is None:
            item['qa_prob'] = self.qa_prob[index]
        item['qa_target'] = self.indexed_datasets['label'][index]

        return item

    def get_text(self, index):
        return self.indexed_datasets['text'][index].reshape(self.num_choices, -1)

    def get_target(self, index):
        if self.saliency_mode == 'coarse':
            target = torch.cat(
                [self.indexed_datasets[self.dataset_keys['saliency_results'][key]][index] for key in
                 self.choice_keys]
            ).flatten()

        elif self.saliency_mode == 'fine':
            assert self.saliency_heuristic in ['gst', 'topk', 'ratio', 'random', 'qa', "random_all", "pos_only_binary",
                                               'raw']
            target = [self.indexed_datasets[self.dataset_keys['saliency_results'][key]][index] for
                      key in self.choice_keys]

            if self.criterion == 'KL_loss':
                target_flat = torch.cat(target)
                return target, target_flat
            else:
                target = torch.cat(target)
                return target

        elif self.saliency_mode == 'hybrid':
            coarse_target = torch.cat(
                [self.indexed_datasets[self.dataset_keys['coarse_saliency_results'][key]][index] for key in
                 self.choice_keys]
            ).flatten()

            fine_target = [self.indexed_datasets[self.dataset_keys['fine_saliency_results'][key]][index] for
                           key in self.choice_keys]

            if self.criterion == 'KL_loss':
                fine_target_flat = torch.cat(fine_target)
                return coarse_target, fine_target, fine_target_flat
            else:
                fine_target = torch.cat(fine_target)
                return coarse_target, fine_target

        return target

    def get_pathgen_data(self, item, index):
        item['path_emb'] = self.path_emb[index]
        return item

    def get_rn_data(self, item, index):
        qa_ids = self.indexed_datasets['qa'][index].reshape(self.num_choices, self.max_tuple_num, 2)
        rel_ids = self.indexed_datasets['rel'][index].reshape(self.num_choices, self.max_tuple_num)
        num_tuples = self.indexed_datasets['num_tuples'][index]

        item['qa_ids'] = qa_ids
        item['rel_ids'] = rel_ids

        for i in range(self.num_choices):
            num_tuples[i] += num_tuples[i] == 0
        item['num_tuples'] = num_tuples

        return item

    def get_mhgrn_data(self, item, index):
        adj_len = self.indexed_datasets['adj_len'][index]
        concept_ids = self.indexed_datasets['concept'][index].reshape(self.num_choices, self.max_node_num)
        node_type_ids = self.indexed_datasets['node_type'][index].reshape(self.num_choices, self.max_node_num)
        adj_tensor = self.adj_empty.clone()
        adj = [self.indexed_datasets['adj_{}'.format(key)][index].reshape(3, -1) for key in self.choice_keys]
        for choice_id, (rel, concept1, concept2) in enumerate(adj):
            adj_tensor[choice_id, rel, concept1, concept2] = 1.

        item['concept_ids'] = concept_ids
        item['node_type_ids'] = node_type_ids
        item['adj_len'] = adj_len
        item['adj'] = adj_tensor

        return item

    def collater(self, instances):
        batch_size = len(instances)
        if batch_size == 0:
            return None

        texts = [instance['text'] for instance in instances]
        indices = [instance['index'] for instance in instances]
        batch = {
            'text': torch.cat(texts),
            'index': torch.LongTensor(indices),
            'size': batch_size,
        }

        if self.saliency_mode == 'hybrid':
            coarse_sal_targets = [instance['coarse_sal_target'] for instance in instances]
            batch['coarse_sal_target'] = torch.cat(coarse_sal_targets).long()

            fine_sal_targets = [instance['fine_sal_target'] for instance in instances]
            if self.criterion == 'KL_loss':
                batch['fine_sal_target_flat'] = torch.cat(
                    [instance['fine_sal_target_flat'] for instance in instances]).long()
                fine_sal_targets = list(itertools.chain.from_iterable(fine_sal_targets))
                if self.graph_encoder == 'mhgrn':
                    max_graph_size = self.max_node_num
                elif self.graph_encoder in ['rn', 'pathgen']:
                    max_graph_size = self.max_tuple_num
                padded_fine_sal_targets = pad_sequence(fine_sal_targets, batch_first=True)
                padded_fine_sal_targets = torch.cat(
                    (padded_fine_sal_targets,
                     torch.zeros(padded_fine_sal_targets.shape[0], max_graph_size - padded_fine_sal_targets.shape[1])),
                    dim=1)
                batch_fine_sal_targets = -self.args.attn_bound * torch.ones_like(padded_fine_sal_targets)
                batch['fine_sal_target'] = torch.masked_fill(batch_fine_sal_targets, padded_fine_sal_targets.bool(),
                                                             self.args.attn_bound)
            else:
                batch['fine_sal_target'] = torch.cat(fine_sal_targets).long() if self.target_type in ['cls',
                                                                                                      'oreg'] else torch.cat(
                    fine_sal_targets).float()

            batch['coarse_pos_weight'] = self.coarse_pos_weight
            batch['fine_pos_weight'] = self.fine_pos_weight

        else:
            targets = [instance['target'] for instance in instances]

            if self.criterion == 'KL_loss':
                flat_target = torch.cat([instance['target_flat'] for instance in instances])
                if self.saliency_heuristic == "raw":
                    # float target
                    batch['target_flat'] = flat_target.double()
                else:  # for non-raw saliency target, use long
                    batch['target_flat'] = flat_target.long()
                targets = list(itertools.chain.from_iterable(targets))
                if self.graph_encoder == 'mhgrn':
                    max_graph_size = self.max_node_num
                elif self.graph_encoder in ['rn', 'pathgen']:
                    max_graph_size = self.max_tuple_num
                padded_targets = pad_sequence(targets, batch_first=True)
                padded_targets = torch.cat(
                    (padded_targets, torch.zeros(padded_targets.shape[0], max_graph_size - padded_targets.shape[1])),
                    dim=1)
                batch_targets = -self.args.attn_bound * torch.ones_like(padded_targets)
                batch['target'] = torch.masked_fill(batch_targets, padded_targets.bool(), self.args.attn_bound)
            else:
                batch['target'] = torch.cat(targets).long() if self.target_type in ['cls', 'oreg'] else torch.cat(
                    targets).float()

            batch['pos_weight'] = self.pos_weight

        if self.sal_gnn_layer_type == 'mhgrn':
            batch['concept_ids'] = torch.stack([instance['concept_ids'] for instance in instances]).reshape(-1,
                                                                                                            self.max_node_num)
            batch['node_type_ids'] = torch.stack([instance['node_type_ids'] for instance in instances]).reshape(-1,
                                                                                                                self.max_node_num)
            batch['adj_len'] = torch.stack([instance['adj_len'] for instance in instances]).reshape(-1)
            batch['adj'] = torch.stack([instance['adj'] for instance in instances]).reshape(-1, self.num_rels - 1,
                                                                                            self.max_node_num,
                                                                                            self.max_node_num)
        elif self.sal_gnn_layer_type == 'pathgen':
            batch['qa_ids'] = torch.stack([instance['qa_ids'] for instance in instances]).reshape(-1,
                                                                                                  self.max_tuple_num, 2)
            batch['rel_ids'] = torch.stack([instance['rel_ids'] for instance in instances]).reshape(-1,
                                                                                                    self.max_tuple_num)
            batch['num_tuples'] = torch.stack([instance['num_tuples'] for instance in instances]).reshape(-1)
            batch['path_emb'] = torch.stack([instance['path_emb'] for instance in instances])
        elif self.sal_gnn_layer_type == 'rn':
            batch['qa_ids'] = torch.stack([instance['qa_ids'] for instance in instances]).reshape(-1,
                                                                                                  self.max_tuple_num, 2)
            batch['rel_ids'] = torch.stack([instance['rel_ids'] for instance in instances]).reshape(-1,
                                                                                                    self.max_tuple_num)
            batch['num_tuples'] = torch.stack([instance['num_tuples'] for instance in instances]).reshape(-1)
        else:
            raise NotImplemented

        batch['qa_target'] = torch.cat([instance['qa_target'] for instance in instances]).long()
        if not self.qa_prob is None:
            batch['qa_prob'] = torch.Tensor([instance['qa_prob'] for instance in instances])

        return batch
