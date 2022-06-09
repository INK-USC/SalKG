from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
	AdamW,
	get_linear_schedule_with_warmup
)

from constants import NUM_CHOICES
from layers import CustomizedEmbedding
from mhgrn import GraphRelationNetModel
from optimization import RAdam
from pathgen import LMRelationNetModel
from rn import RelationNetModel
from text_encoder import TextEncoder
from utils import cal_2hop_rel_emb, freeze_net, unfreeze_net

graph_encoder_dict = {
    'rn': RelationNetModel,
    'mhgrn': GraphRelationNetModel,
    'pathgen': LMRelationNetModel
}


class QAModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.text_encoder = TextEncoder(args)
        sent_dim = self.text_encoder.model.config.hidden_size
        if args.graph_encoder != 'no_kg':
            # KG model
            assert args.graph_encoder in ['rn', 'mhgrn', 'pathgen']
            concept_emb, rel_emb = self.initalize_embeddings()
            self.graph_encoder = graph_encoder_dict[args.graph_encoder](args, concept_emb, rel_emb, sent_dim)

        # NO KG variants
        elif args.graph_encoder == 'no_kg' and args.no_kg_emb == 'no_emb':
            self.qa_classifier = nn.Linear(args.text_out_dim, 1)
        elif args.graph_encoder == 'no_kg' and args.no_kg_emb == 'zero_emb':
            self.no_kg_emb = torch.zeros(1, args.no_kg_emb_dim)
            self.qa_classifier = nn.Linear(args.no_kg_emb_dim + args.text_out_dim, 1)
        elif args.graph_encoder == 'no_kg' and args.no_kg_emb == 'learned_emb':
            self.no_kg_emb = nn.Parameter(torch.randn(1, args.no_kg_emb_dim))
            self.qa_classifier = nn.Linear(args.no_kg_emb_dim + args.text_out_dim, 1)

    def forward(self, batch: dict):
        """
        text: (bsz * num_choices, seq_len) text cat with answer
        target: (bsz, ) answer choice key
        index: (bsz, ) which question (index in __getitem__)
        size: scalar = bsz
        if MHGRN:
            concept_ids: (bsz * num_choices, n_node=200) first k col are concepts that has relations
            node_type_ids: (bsz * num_choices, n_node=200)
              indicate which among concept_ids are q/a/others
              0 == question node; 1 == answer node: 2 == intermediate node
            adj_len: (bsz * num_choices, ) i.e. k in "first k cols"
            adj: (bsz * num_choices, n_relation=34, n_node=200, n_node=200)
              adj[r, s, t] = 1 => node s & t has relation r
        if Pathgen:
            qa_ids: (bsz * num_choices, max_tuple_num=200, 2)
              first k cols are 2 q & a ids (q & a has relations within 2 hops)
            rel_ids: (bsz * num_choices, max_tuple_num=200)
              first k cols are relations between q & a
              if 1 relation, = rel id; if 2 = 34 + rel0 * 34 + rel1
            num_tuples: (bsz * num_choices, ) i.e. k in "first k cols"
            path_emb: (bsz, num_choices, num_choices, 768)
        if RN:
            'qa_ids', 'rel_ids', 'num_tuples'
        ----------------------------------------
        output logits (bsz, num_choices)
        """

        # Encode text (bsz * num_choices, seq_len) -> (bsz * num_choices, hidden_dim)
        text_emb = self.text_encoder(batch['text'])

        # Predict QA scores
        if self.args.graph_encoder != 'no_kg':
            # (bsz * num_choices, 1)
            logits, _, sal_input = self.graph_encoder(batch, text_emb)

        elif self.args.graph_encoder == 'no_kg':
            # based on no_kg_emb, output logits (bsz * num_choices, 1)
            if self.args.no_kg_emb == 'no_emb':
                cls_emb = text_emb
            elif self.args.no_kg_emb in ['zero_emb', 'learned_emb']:
                no_kg_emb = self.no_kg_emb.expand(batch['size'] * NUM_CHOICES[self.args.dataset], -1).to(
                    text_emb.device)
                cls_emb = torch.cat((no_kg_emb, text_emb), dim=-1)
            logits = self.qa_classifier(cls_emb)
            sal_input = None
        else:
            raise NotImplementedError

        return logits.reshape(batch['size'], -1), sal_input

    def initalize_embeddings(self):
        # (799273, 1024)
        concept_emb_ = torch.from_numpy(np.load(self.args.ent_emb_path))
        concept_num, concept_dim = concept_emb_.shape

        concept_emb = CustomizedEmbedding(
            concept_num=concept_num,
            concept_out_dim=concept_dim,
            use_contextualized=False,
            concept_in_dim=concept_dim,
            pretrained_concept_emb=concept_emb_,
            freeze_ent_emb=self.args.graph_freeze_ent_emb
        )

        if self.args.graph_encoder in ['rn', 'pathgen']:
            rel_emb_ = torch.from_numpy(np.load(self.args.rel_emb_path))
            rel_emb_ = np.concatenate((rel_emb_, -rel_emb_), 0)
            rel_emb_ = cal_2hop_rel_emb(rel_emb_)
            rel_emb_ = torch.tensor(rel_emb_)
            relation_num, relation_dim = rel_emb_.shape
            rel_emb = nn.Embedding(relation_num, relation_dim)
            rel_emb.weight.data.copy_(rel_emb_)
        elif self.args.graph_encoder == 'mhgrn':
            rel_emb = None
        else:
            raise NotImplementedError

        return concept_emb, rel_emb

    def calc_loss(self, preds, targets):
        return F.cross_entropy(preds, targets)

    def calc_acc(self, preds, targets):
        return 100 * (preds == targets).float().mean()

    def training_step(self, batch, batch_idx):
        # freeze encoder for initial few epochs based on args.freeze_epochs
        if self.current_epoch < self.args.freeze_epochs:
            freeze_net(self.text_encoder)
        else:
            unfreeze_net(self.text_encoder)
        # (bsz, num_choices),  sal_input ignored since this is only used in saving saliency scores
        logits, _ = self(batch)
        preds = torch.argmax(logits, dim=1)
        # (bsz, )
        targets = batch['target']
        loss = self.calc_loss(logits, targets)
        acc = self.calc_acc(preds, targets)

        self.log('train_loss_step', loss, prog_bar=True)
        self.log('train_acc_step', acc, prog_bar=True)
        return {'loss': loss, 'size': torch.tensor(batch['size']).cuda(), 'acc': acc}

    def training_epoch_end(self, outputs):
        weights = torch.stack([x['size'] for x in outputs])
        weights = weights / weights.sum()

        loss = torch.stack([x['loss'] for x in outputs])
        loss_testing = (loss * weights).sum()
        self.log('train_loss_epoch', loss_testing.item())

        acc = torch.stack([x['acc'] for x in outputs])
        acc = (acc * weights).sum()
        self.log('train_acc_epoch', acc.item())

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits, _ = self(batch)
        preds = torch.argmax(logits, dim=1)
        targets = batch['target']
        loss = self.calc_loss(logits, targets)
        acc = self.calc_acc(preds, targets)

        self.log('valid_loss_step', loss, prog_bar=True)
        self.log('valid_acc_step', acc, prog_bar=True)
        return {'loss': loss, 'size': torch.tensor(batch['size']).cuda(), 'acc': acc}

    def validation_epoch_end(self, outputs):
        weights = torch.stack([x['size'] for x in outputs])
        weights = weights / weights.sum()

        loss = torch.stack([x['loss'] for x in outputs])
        loss = (loss * weights).sum()
        self.log('valid_loss_epoch', loss.item())

        acc = torch.stack([x['acc'] for x in outputs])
        acc = (acc * weights).sum()
        self.log('valid_acc_epoch', acc.item())

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logits, _ = self(batch)
        preds = torch.argmax(logits, dim=1)
        targets = batch['target']
        loss = self.calc_loss(logits, targets)
        acc = self.calc_acc(preds, targets)

        self.log('test_loss_step', loss, prog_bar=True)
        self.log('test_acc_step', acc, prog_bar=True)

        return {'loss': loss, 'preds': preds, 'targets': targets}

    def test_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs])
        targets = torch.cat([x['targets'] for x in outputs])
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        acc = self.calc_acc(preds, targets)

        self.log('test_loss_epoch', loss.item())
        self.log('test_acc_epoch', acc)

    def setup(self, stage):
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = (
                    (len(train_loader.dataset) // (self.args.train_batch_size * max(1, self.args.gpus)))
                    // self.args.accumulate_grad_batches
                    * float(self.args.max_epochs)
            )

    def configure_optimizers(self):
        'Prepare optimizer and schedule (linear warmup and decay)'
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.text_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay,
                'lr': self.args.text_lr
            },
            {
                'params': [p for n, p in self.text_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.args.text_lr
            }
        ]

        if self.args.graph_encoder != 'no_kg':
            optimizer_grouped_parameters += [
                {
                    'params': [p for n, p in self.graph_encoder.named_parameters() if
                               not any(nd in n for nd in no_decay)],
                    'weight_decay': self.args.weight_decay,
                    'lr': self.args.graph_lr
                },
                {
                    'params': [p for n, p in self.graph_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': self.args.graph_lr
                }
            ]
        else:
            optimizer_grouped_parameters += [
                {
                    'params': [p for n, p in self.qa_classifier.named_parameters() if
                               not any(nd in n for nd in no_decay)],
                    'weight_decay': self.args.weight_decay,
                    'lr': self.args.cls_lr
                },
                {
                    'params': [p for n, p in self.qa_classifier.named_parameters() if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': self.args.cls_lr
                }
            ]

        if self.args.optimizer == 'adamw':
            optimizer = AdamW(optimizer_grouped_parameters)
        elif self.args.optimizer == 'radam':
            optimizer = RAdam(optimizer_grouped_parameters)
        else:
            raise NotImplementedError

        if self.args.lr_scheduler == 'linear_with_warmup':
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_updates,
                                                        num_training_steps=self.total_steps)
            scheduler = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        elif self.args.lr_scheduler == 'fixed':
            return [optimizer]
        else:
            raise NotImplementedError

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--optimizer', default='radam', type=str, choices=['adamw', 'radam'])
        parser.add_argument('--no_xavier', default=False, action='store_true')
        parser.add_argument('--use_magnitude_grad', default=False, action='store_true')
        parser.add_argument('--text_lr', default=2e-5, type=float)
        parser.add_argument('--graph_lr', default=1e-3, type=float)
        parser.add_argument('--cls_lr', default=1e-3, type=float)
        parser.add_argument('--adam_epsilon', default=1e-8, type=float)
        parser.add_argument('--warmup_updates', default=0, type=int)
        parser.add_argument('--weight_decay', default=0.0, type=float)
        parser.add_argument('--lr_scheduler', default='fixed', type=str, choices=['fixed', 'linear_with_warmup'])
        parser.add_argument('--arch', default='albert_xxlarge_v2', type=str,
                            choices=['albert_xxlarge_v2', 'roberta_base', 'roberta_large'])

        parser.add_argument('--graph_encoder', default='mhgrn', type=str, choices=['rn', 'mhgrn', 'no_kg', 'pathgen'])
        parser.add_argument('--graph_init_range', default=0.02, type=float)
        parser.add_argument('--graph_init_rn', action='store_true')
        parser.add_argument('--graph_init_identity', action='store_true')
        parser.add_argument('--graph_k', default=2, type=int)
        parser.add_argument('--graph_n_type', default=3, type=int)
        parser.add_argument('--graph_num_relation', default=34, type=int)
        parser.add_argument('--graph_num_basis', default=0, type=int)
        parser.add_argument('--graph_gnn_layer_num', default=1, type=int)
        parser.add_argument('--graph_diag_decompose', action='store_true')
        parser.add_argument('--graph_att_head_num', default=2, type=int)
        parser.add_argument('--graph_att_dim', default=50, type=int)
        parser.add_argument('--graph_att_layer_num', default=1, type=int)
        parser.add_argument('--graph_dropouti', default=0.1, type=float)
        parser.add_argument('--graph_dropoutg', default=0.2, type=float)
        parser.add_argument('--graph_dropoutf', default=0.2, type=float)
        parser.add_argument('--graph_eps', default=1e-15, type=float)
        parser.add_argument('--graph_fc_dim', default=200, type=int)
        parser.add_argument('--graph_fc_layer_num', default=0, type=int)
        parser.add_argument('--graph_freeze_ent_emb', action='store_true')

        parser.add_argument('--graph_mlp_dim', default=128, type=int)
        parser.add_argument('--graph_mlp_layer_num', default=2, type=int)
        parser.add_argument('--graph_dropoutm', default=0.3, type=float)
        parser.add_argument('--graph_pool', default='multihead_pool', type=str,
                            choices=['none', 'multihead_pool', 'att_pool'])
        parser.add_argument('--graph_emb_scale', default=1.0, type=float)

        parser.add_argument('--ent_emb_dim', default=1024, type=int)
        parser.add_argument('--rel_emb_dim', default=100, type=int)

        parser.add_argument('--text_in_dim', default=4096, type=int)
        parser.add_argument('--text_out_dim', default=4096, type=int)
        parser.add_argument('--text_hidden_dim', default=4096, type=int)
        parser.add_argument('--text_hidden_layers', default=1, type=int)
        parser.add_argument('--text_dropout', action='store_true')
        parser.add_argument('--text_layer_norm', action='store_true')
        parser.add_argument('--text_encoder_head', default='bos_token_mlp', type=str)
        parser.add_argument('--encoder_pooler', default='cls', type=str, choices=['cls', 'mean'])

        parser.add_argument('--no_kg_emb', default='no_emb', type=str, choices=['no_emb', 'zero_emb', 'learned_emb'])
        parser.add_argument('--no_kg_emb_dim', default=1024, type=int)
        parser.add_argument('--graph_emb_dim', default=1024, type=int)

        parser.add_argument('--input_dim_gpt', default=768, type=int)

        parser.add_argument('--aristo_path',
                            default='../data/mhgrn_data/obqa/roberta_mc_arcplus-try34-model/weights.th', type=str)

        return parser