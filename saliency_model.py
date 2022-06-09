import csv
import os
import random
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_curve
from torch import nn
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from constants import NUM_CHOICES
from layers import CustomizedEmbedding, MLP_factory, MultiheadAttPoolLayer
from mhgrn import GraphRelationNetModel
from optimization import RAdam
from pathgen import LMRelationNetModel
from rn import RelationNetModel
from text_encoder import TextEncoder
from utils import cal_2hop_rel_emb, freeze_net, unfreeze_net

gnn_dict = {
    'mhgrn': GraphRelationNetModel,
    'rn': RelationNetModel,
    'pathgen': LMRelationNetModel
}


class SaliencyModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        assert args.graph_encoder in ['rn', 'mhgrn', 'pathgen']
        assert args.saliency_mode in ['coarse', 'fine', 'hybrid']
        assert args.sal_gnn_layer_type in ['mhgrn', 'rn', 'pathgen']
        assert args.target_type in ['cls', 'reg']
        assert args.graph_encoder == args.sal_gnn_layer_type

        self.args = args
        self.num_choices = NUM_CHOICES[args.dataset]
        self.text_encoder = TextEncoder(args)
        sent_dim = self.text_encoder.model.config.hidden_size

        if self.args.criterion in ['bce_loss', 'KL_loss']:
            self.best_valid_f1 = -1
            self.best_valid_f1_thresh = -1

        if args.saliency_mode == 'hybrid':
            coarse_concept_emb, coarse_rel_emb = self.initalize_embeddings()
            self.coarse_saliency_gnn = gnn_dict[args.sal_gnn_layer_type](args, coarse_concept_emb, coarse_rel_emb,
                                                                         sent_dim)
            if getattr(args, "fine_checkpoint_path", None):
                assert os.path.exists(args.fine_checkpoint_path)
                print("loading checkpoint", args.fine_checkpoint_path, flush=True)
                # trainable fine saliency gnn with pretrained weights
                self.fine_saliency_gnn = type(self).load_from_checkpoint(args.fine_checkpoint_path).saliency_gnn
            else:
                fine_concept_emb, fine_rel_emb = self.initalize_embeddings()
                self.fine_saliency_gnn = gnn_dict[args.sal_gnn_layer_type](args, fine_concept_emb, fine_rel_emb,
                                                                           sent_dim)
        else:
            concept_emb, rel_emb = self.initalize_embeddings()
            self.saliency_gnn = gnn_dict[args.sal_gnn_layer_type](args, concept_emb, rel_emb, sent_dim)

        io_factor = 1
        input_dim = args.sal_att_input_dim if args.sal_att_pooling else io_factor * args.sal_gnn_io_dim + args.text_out_dim

        if args.saliency_mode == 'hybrid':
            sal_cls_layer_sizes = [
                [input_dim, 1],
                [args.sal_cls_hidden_dim, args.sal_cls_hidden_layers],
                [2, 1]
            ]
        elif args.criterion in ['bce_loss', 'mse_loss', 'mae_loss', 'KL_loss']:
            sal_cls_layer_sizes = [
                [input_dim, 1],
                [args.sal_cls_hidden_dim, args.sal_cls_hidden_layers],
                [1, 1]
            ]
        elif args.criterion in ['ce_loss']:
            assert args.sal_num_classes >= 2
            num_classes = args.sal_num_classes
            sal_cls_layer_sizes = [
                [input_dim, 1],
                [args.sal_cls_hidden_dim, args.sal_cls_hidden_layers],
                [num_classes, 1]
            ]
        else:
            raise NotImplementedError

        if not (args.pruned_qa and args.saliency_mode == "fine"):
            # only do MLP to get sal output when not pruned fine
            self.saliency_predictor = MLP_factory(
                sal_cls_layer_sizes,
                dropout=args.sal_cls_dropout,
                layer_norm=args.sal_cls_layer_norm,
                coral=(args.criterion == 'coral_loss')
            )

            if args.pruned_qa and args.coarse_model == 'graph_emb':
                qa_cls_layer_sizes = [
                    [input_dim, 1],
                    [args.sal_cls_hidden_dim, args.sal_cls_hidden_layers],
                    [1, 1]
                ]
                self.qa_predictor = MLP_factory(
                    qa_cls_layer_sizes,
                    dropout=args.sal_cls_dropout,
                    layer_norm=args.sal_cls_layer_norm
                )

        if args.sal_att_pooling:
            self.graph2att = nn.Linear(io_factor * args.sal_gnn_io_dim + args.text_out_dim, args.sal_att_input_dim)
            self.text2att = nn.Linear(args.text_out_dim, args.sal_att_input_dim)
            self.attention = MultiheadAttPoolLayer(args.sal_num_att_head, args.sal_att_input_dim,
                                                   args.sal_att_input_dim)

        if args.sal_cls_bias in ['ba_set', 'ba_fix']:
            train_lp_path = os.path.join(self.args.lp_dir,
                                         f'train.mhgrn.sal_coarse_occl_target_lp_cls{self.args.sal_num_classes}.npy')
            valid_lp_path = os.path.join(self.args.lp_dir,
                                         f'valid.mhgrn.sal_coarse_occl_target_lp_cls{self.args.sal_num_classes}.npy')
            self.log_prior = {
                'train': torch.from_numpy(np.load(train_lp_path)).float().cuda(),
                'valid': torch.from_numpy(np.load(valid_lp_path)).float().cuda()
            }
            if args.sal_cls_bias == 'ba_fix':
                self.saliency_predictor[-1].bias.requires_grad = False
        if args.coarse_model == 'graph_emb':
            if args.no_kg_emb == 'learned':
                self.no_kg_emb = nn.Parameter(torch.randn(1, args.sal_gnn_io_dim))
            elif args.no_kg_emb == 'random':
                self.no_kg_emb = torch.randn(1, args.sal_gnn_io_dim)
            elif args.no_kg_emb == 'zero':
                self.no_kg_emb = torch.zeros(1, args.sal_gnn_io_dim)

    def initalize_embeddings(self):
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
        concept_emb.emb.weight.data.copy_(concept_emb_)

        if self.args.graph_encoder == 'rn':
            rel_emb_ = torch.from_numpy(np.load(self.args.rel_emb_path))
            rel_emb_ = np.concatenate((rel_emb_, -rel_emb_), 0)
            rel_emb_ = cal_2hop_rel_emb(rel_emb_)
            rel_emb_ = torch.tensor(rel_emb_)
            relation_num, relation_dim = rel_emb_.shape
            rel_emb = nn.Embedding(relation_num, relation_dim)
            rel_emb.weight.data.copy_(rel_emb_)
        elif self.args.graph_encoder == 'mhgrn':
            rel_emb = None
        elif self.args.graph_encoder == 'pathgen':
            rel_emb_ = torch.from_numpy(np.load(self.args.rel_emb_path))
            rel_emb_ = np.concatenate((rel_emb_, -rel_emb_), 0)
            rel_emb_ = cal_2hop_rel_emb(rel_emb_)
            rel_emb_ = torch.tensor(rel_emb_)
            relation_num, relation_dim = rel_emb_.shape
            rel_emb = nn.Embedding(relation_num, relation_dim)
            rel_emb.weight.data.copy_(rel_emb_)

        return concept_emb, rel_emb

    def forward(self, batch):

        # Encode text
        text_enc, _ = self.text_encoder(batch['text'])

        # Encode graph
        if self.args.saliency_mode == "coarse":
            unit_feat, mask, graph_feat, _ = self.saliency_gnn(batch, text_enc, bypass_logits=True)
            concat_feat = torch.cat((graph_feat, text_enc), dim=-1)

            # Predict logits
            if self.args.sal_att_pooling:
                g_feat = self.graph2att(concat_feat)
                t_feat = self.text2att(text_enc)
                pooled_vecs, scores = self.attention(t_feat, torch.stack((t_feat, g_feat), dim=1))
                logits = self.saliency_predictor(pooled_vecs)
            else:
                logits = self.saliency_predictor(concat_feat)

            if self.args.coarse_model == 'graph_emb':
                sal_probs = F.softmax(logits.reshape(-1, 2), dim=-1)
                no_kg_feat = sal_probs[:, 0].unsqueeze(1).expand(-1, self.args.sal_gnn_io_dim) * self.no_kg_emb.expand(
                    batch['size'] * self.num_choices, -1).cuda()
                kg_feat = sal_probs[:, 1].unsqueeze(1).expand(-1, self.args.sal_gnn_io_dim) * graph_feat
                qa_graph_feat = no_kg_feat + kg_feat
                qa_concat_feat = torch.cat((qa_graph_feat, text_enc), dim=-1)
                qa_logits = self.qa_predictor(qa_concat_feat).reshape(batch['size'], self.num_choices)

                output_dict = {'qa': qa_logits}
                output_dict['sal'] = logits

                return output_dict

        elif self.args.saliency_mode == "fine":
            # pool_attn (bsz * head * num_choice, max_len=200)
            # mask (bsz * num_choice, max_len)
            unit_feat, mask, graph_feat, pool_attn, qa_logits = self.saliency_gnn(batch, text_enc, end2end=True,
                                                                                  return_raw_attn=True)
            qa_logits = qa_logits.view(batch["size"], -1)

            pool_attn = pool_attn[1]  # raw attention scores (before softmax and dropout)
            pool_attn = torch.clamp(pool_attn, min=-self.args.attn_bound,
                                    max=self.args.attn_bound)  # clamp raw attention scores to be in [-ATTN_BOUND, ATTN_BOUND] (esp. to deal with -inf scores)
            # (bsz*num_choice, #heads, max_len)
            pool_attn = pool_attn.view(unit_feat.shape[0], self.args.graph_att_head_num, pool_attn.shape[1])
            # select some k heads from all heads
            if self.args.attn_agg_k != "none":
                # if random_step, self.att_agg_k = k aka #select heads
                # otherwise, self.att_agg_k = list of k index for selection
                ks = random.sample(range(self.args.graph_att_head_num),
                                   self.att_agg_k) if self.att_agg_k_mode == "random_step" \
                    else self.att_agg_k
                ks = torch.tensor(ks, device=self.device)
                pool_attn = pool_attn.index_select(1, ks)

            mask_nonzero = torch.nonzero(mask)
            unit2batch = mask_nonzero[:, 0]
            unit_mask = unit_feat.shape[1] * mask_nonzero[:, 0] + mask_nonzero[:, 1]
            if self.args.attn_head_agg == "avg_head":
                # avg over heads
                pool_attn = pool_attn.mean(1)
                # (#valid, )
                pool_attn_flat = torch.index_select(pool_attn.view(-1, 1), 0, unit_mask).squeeze()  # (224,)
            elif self.args.attn_head_agg == "avg_loss":

                # pool_attn & pool_attn_flat keep #heads dim
                pool_attn_flat = pool_attn.view(pool_attn.size(1), -1)
                # (#head, #valid)
                pool_attn_flat = pool_attn_flat.index_select(1, unit_mask)
            else:
                raise NotImplementedError

            if self.args.pruned_qa:  # joint training
                if self.args.save_saliency:
                    self.save_predictions(batch, qa_logits)
                return qa_logits, pool_attn, pool_attn_flat, mask

            else:  # pipeline training
                unit_feat = torch.index_select(unit_feat.view(-1, unit_feat.shape[-1]), 0, unit_mask)
                unit_counts = torch.bincount(unit2batch, minlength=self.num_choices * batch['size'])
                graph_feat = torch.repeat_interleave(graph_feat, unit_counts, dim=0)
                text_enc = torch.repeat_interleave(text_enc, unit_counts, dim=0)
                concat_feat = torch.cat((unit_feat, graph_feat, text_enc), dim=-1)

                logits = self.saliency_predictor(concat_feat)

        elif self.args.saliency_mode == "hybrid":
            # coarse
            _, _, graph_feat, _ = self.coarse_saliency_gnn(batch, text_enc, bypass_logits=True)
            concat_feat = torch.cat((graph_feat, text_enc), dim=-1)
            coarse_logits = self.saliency_predictor(concat_feat)

            # fine
            unit_feat, mask, _, pool_attn, fine_logits = self.fine_saliency_gnn(batch, text_enc, end2end=True,
                                                                                return_raw_attn=True)
            pool_attn = pool_attn[1]  # raw attention scores (before softmax and dropout)
            pool_attn = torch.clamp(pool_attn, min=-self.args.attn_bound,
                                    max=self.args.attn_bound)  # clamp raw attention scores to be in [-ATTN_BOUND, ATTN_BOUND] (esp. to deal with -inf scores)
            pool_attn = pool_attn.view(unit_feat.shape[0], self.args.graph_att_head_num, pool_attn.shape[1]).mean(1)
            mask_nonzero = torch.nonzero(mask)
            unit2batch = mask_nonzero[:, 0]
            unit_mask = unit_feat.shape[1] * mask_nonzero[:, 0] + mask_nonzero[:, 1]
            pool_attn_flat = torch.index_select(pool_attn.view(-1, 1), 0, unit_mask).squeeze()

            return coarse_logits, fine_logits, pool_attn, pool_attn_flat, mask

        if self.args.criterion in ['bce_loss', 'mse_loss', 'mae_loss']:
            logits = logits.flatten()

        if self.args.save_saliency:
            self.save_saliency_scores(batch, logits)

        return logits

    def save_predictions(self, batch, logits):
        logits = logits.reshape(batch['size'], -1)
        probs = F.softmax(logits, dim=-1)
        for i in range(batch['size']):
            cur_index = batch['index'][i].item()
            cur_target = batch['qa_target'][i].item()
            cur_pred = [x.item() for x in probs[i]]
            cur_data = [cur_index, cur_target] + cur_pred
            with open(self.saliency_path, 'a') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(cur_data)

    def save_saliency_scores(self, batch, logits):

        if self.args.saliency_mode == 'coarse':
            if self.args.target_type == 'cls':
                if self.args.criterion == 'ce_loss':
                    probs = F.softmax(logits, dim=-1)
                    preds = torch.argmax(probs, dim=1)
                elif self.args.criterion == 'bce_loss':
                    preds = torch.sigmoid(logits)
                preds = preds.reshape(batch['size'], -1)
            elif self.args.target_type == 'reg':
                preds = logits.reshape(batch['size'], -1)

            for i in range(batch['size']):
                cur_index = batch['index'][i].item()
                cur_pred = preds[i].tolist()
                cur_data = [cur_index] + cur_pred
                with open(self.saliency_path, 'a') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow(cur_data)

        elif self.args.saliency_mode == 'fine':
            probs = F.softmax(logits, dim=-1)
            preds = probs[:, 1]
            if self.args.graph_encoder == 'mhgrn':
                prev_idx = 0
                cum_adj_len = torch.cumsum(batch['adj_len'], dim=0).reshape(batch['size'],
                                                                            NUM_CHOICES[self.args.dataset])
                for i in range(batch['size']):
                    for j in range(NUM_CHOICES[self.args.dataset]):
                        data = preds[prev_idx:cum_adj_len[i, j]].tolist()
                        prev_idx = cum_adj_len[i, j]
                        if type(data) != list:
                            # edge case when len(all_ids) == 1 (tolist is buggy and returns just int in that case)
                            data = [data]
                        cur_data = [self.batch_counter, j] + data
                        with open(self.saliency_path, 'a') as f:
                            writer = csv.writer(f, delimiter=',')
                            writer.writerow(cur_data)
                    self.batch_counter += 1

            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def calc_loss(self, outputs, targets, pos_weight, mask=None, graph_sizes=None, criterion=None):
        if criterion == None:
            criterion = self.args.criterion
        if self.args.attn_head_agg == "avg_loss":
            assert criterion in ["KL_loss", "mse_loss"]
            if criterion == "mse_loss":
                # (#heads, #valid) outputs
                assert outputs.ndim == 2 and targets.ndim == 1
                # repeat target (#valid, ) to (#heads, #valid)
                targets = targets.repeat(outputs.size(0), 1)
            else:
                # (bsz*#choice, #heads, max_len) outputs
                # (bsz*#choice, max_len) targets
                # (bsz*#choice, max_len) mask
                # (bsz*#choice, ) graph_size
                assert outputs.ndim == 3 and targets.ndim == 2 and mask is not None and graph_sizes is not None
                # either graph_att_head_num or k if specify avg_loss_k
                num_heads = outputs.size(1)
                # resize to (bsz*#choice*#heads, max_len)
                outputs = outputs.view(outputs.size(0) * num_heads, outputs.size(2))
                targets = targets.repeat((num_heads, 1))
                mask = mask.repeat((num_heads, 1))
                graph_sizes = graph_sizes.repeat(num_heads)

        if criterion in ['bce_loss']:
            if pos_weight > 0 and pos_weight != 1:
                weight = torch.ones(len(targets)).cuda()
                weight[torch.nonzero(targets)] = pos_weight
                loss = F.binary_cross_entropy_with_logits(outputs, targets.to(torch.float), weight=weight)
            else:
                loss = F.binary_cross_entropy_with_logits(outputs, targets.to(torch.float))

        elif criterion in ['ce_loss']:
            if self.args.sal_num_classes == 2:
                loss = F.cross_entropy(outputs, targets, weight=torch.cuda.FloatTensor([1, pos_weight]))
            else:
                loss = F.cross_entropy(outputs, targets)

        elif criterion in ["KL_loss"]:
            output_distr = F.log_softmax(outputs, dim=1)
            target_distr = F.softmax(targets, dim=1)
            loss = F.kl_div(output_distr, target_distr, reduction='none')
            loss = loss * mask
            loss = torch.mean(loss.sum(1) / graph_sizes)

        elif criterion in ['mse_loss']:
            loss = F.mse_loss(outputs, targets.float().softmax(-1))

        elif criterion in ['mae_loss']:
            loss = F.l1_loss(outputs, targets)


        else:
            raise NotImplementedError

        return loss

    def calc_f1(self, outputs, targets, split):
        if self.args.saliency_heuristic == "raw" or self.args.saliency_method == "ig" or self.args.criterion == "KL_loss":
            # not define in raw score
            return {
                'f1': -1,
                "f1_thresh": -1
            }
        f1_dict = {}
        if self.args.criterion in ['bce_loss', 'KL_loss']:
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            targets = targets.cpu().numpy()
            if split == 'train':
                preds = probs >= 0.5
                f1 = 100 * f1_score(targets, preds, average='binary', zero_division=0)
                f1_thresh = 0.5
            elif split == 'valid':
                precision, recall, thresholds = precision_recall_curve(targets, probs, pos_label=1)
                f1_num = 2 * precision * recall
                f1_den = precision + recall
                invalid_indices = np.argwhere(f1_den == 0 * ~np.isfinite(f1_num) * ~np.isfinite(f1_den))
                f1_num[invalid_indices] = 0
                f1_den[invalid_indices] = 1
                f1_scores = f1_num / f1_den
                f1 = 100 * np.max(f1_scores)
                f1_thresh = thresholds[np.argmax(f1_scores)]
                if f1 > self.best_valid_f1:
                    self.best_valid_f1 = f1
                    self.best_valid_f1_thresh = f1_thresh
            else:
                preds = probs >= self.best_valid_f1_thresh
                f1 = 100 * f1_score(targets, preds, average='binary', zero_division=0)
                f1_thresh = self.best_valid_f1_thresh

            f1_dict['f1'] = f1
            f1_dict['f1_thresh'] = f1_thresh

        elif self.args.criterion in ['ce_loss']:
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            targets = targets.cpu().numpy()
            f1_dict['f1'] = 100 * f1_score(targets, preds, average='binary', zero_division=0)

        else:
            raise NotImplementedError

        return f1_dict

    def calc_acc(self, outputs, targets):
        preds = torch.argmax(outputs, dim=1)
        return 100 * (preds == targets).float().mean()

    def training_step(self, batch, batch_idx):
        sal_loss_weight = self.loss_weight.data
        # freeze encoder for initial few epochs based on args.freeze_epochs
        if self.args.freeze_epochs > 0:
            if self.current_epoch < self.args.freeze_epochs:
                freeze_net(self.text_encoder)
            else:
                unfreeze_net(self.text_encoder)

        if self.args.sal_cls_bias in ['ba_fix']:
            self.saliency_predictor[-1].bias.data = self.log_prior['train']

        outputs = self(batch)
        if self.args.saliency_mode != 'hybrid':
            targets = batch['target']
        else:
            coarse_sal_targets = batch['coarse_sal_target']
            fine_sal_targets = batch['fine_sal_target']
        qa_targets = batch['qa_target']

        if self.args.pruned_qa:
            if self.args.saliency_mode == 'coarse':
                if self.args.coarse_model == 'ensemble':
                    sal_outputs = outputs
                    sal_probs = F.softmax(sal_outputs.reshape(batch['qa_prob'].shape), dim=-1)
                    qa_outputs = torch.sum(sal_probs * batch['qa_prob'], dim=-1)
                elif self.args.coarse_model == 'graph_emb':
                    qa_outputs = outputs['qa']
                    sal_outputs = outputs['sal']

                qa_loss = F.cross_entropy(qa_outputs, qa_targets)
                sal_loss = self.calc_loss(sal_outputs, targets, batch['pos_weight'])

                loss = qa_loss + sal_loss_weight * sal_loss
                qa_acc = self.calc_acc(qa_outputs, qa_targets)

            elif self.args.saliency_mode == 'fine':
                qa_outputs, attn_weights, attn_weights_flat, mask = outputs
                graph_sizes = batch['adj_len'] if self.args.graph_encoder == 'mhgrn' else batch['num_tuples']
                if self.args.criterion in ['bce_loss', 'mse_loss']:
                    attn_weights = attn_weights_flat  # (#valid, ) if avg_head else (#head, #valid)

                if self.args.saliency_heuristic == "raw":
                    # use flatten attn to approximate flatten raw saliency
                    sal_loss = F.kl_div(F.log_softmax(attn_weights_flat.double()), F.softmax(batch["target_flat"]))
                else:
                    sal_loss = self.calc_loss(attn_weights, targets, batch['pos_weight'], mask, graph_sizes)
                qa_loss = F.cross_entropy(qa_outputs, qa_targets)

                loss = qa_loss + sal_loss_weight * sal_loss

                qa_acc = self.calc_acc(qa_outputs, qa_targets)

                # consistent with others config's f1
                outputs = attn_weights_flat
                if self.args.attn_head_agg == "avg_loss":
                    outputs = outputs.mean(dim=1)
                targets = batch['target'] if self.args.criterion in ['bce_loss', 'mse_loss'] else batch['target_flat']

            elif self.args.saliency_mode == 'hybrid':
                coarse_sal_outputs, fine_qa_outputs, attn_weights, attn_weights_flat, mask = outputs

                # coarse
                if self.args.coarse_model == 'ensemble':
                    fine_qa_probs = F.softmax(fine_qa_outputs.reshape(batch['qa_prob'].shape), dim=-1)
                    qa_probs = torch.stack((batch['qa_prob'], fine_qa_probs), dim=-1)
                    coarse_sal_probs = F.softmax(coarse_sal_outputs.reshape(qa_probs.shape), dim=-1)
                    coarse_qa_outputs = torch.sum(coarse_sal_probs * qa_probs, dim=-1)
                else:
                    raise NotImplementedError

                qa_loss = F.cross_entropy(coarse_qa_outputs, qa_targets)
                coarse_sal_loss = self.calc_loss(coarse_sal_outputs, coarse_sal_targets, batch['coarse_pos_weight'],
                                                 criterion='ce_loss')
                coarse_loss = qa_loss + sal_loss_weight * coarse_sal_loss

                # fine
                graph_sizes = batch['adj_len'] if self.args.graph_encoder == 'mhgrn' else batch['num_tuples']
                if self.args.criterion == 'bce_loss':
                    attn_weights = attn_weights_flat

                fine_qa_outputs = fine_qa_outputs.reshape(coarse_qa_outputs.shape)
                fine_qa_loss = F.cross_entropy(fine_qa_outputs, qa_targets)
                fine_sal_loss = self.calc_loss(attn_weights, fine_sal_targets, batch['fine_pos_weight'], mask,
                                               graph_sizes)
                fine_loss = fine_qa_loss + sal_loss_weight * fine_sal_loss
                fine_qa_acc = self.calc_acc(fine_qa_outputs, qa_targets)

                # hybrid
                loss = coarse_loss + self.args.fine_loss_weight * fine_loss
                qa_acc = self.calc_acc(coarse_qa_outputs, qa_targets)

                self.log('train_coarse_sal_loss_step', coarse_sal_loss.detach(), prog_bar=True)

                self.log('train_fine_qa_acc_step', fine_qa_acc, prog_bar=True)
                self.log('train_fine_qa_loss_step', fine_qa_loss.detach(), prog_bar=True)
                self.log('train_fine_sal_loss_step', fine_sal_loss.detach(), prog_bar=True)

            self.log('train_qa_acc_step', qa_acc, prog_bar=True)
            self.log('train_qa_loss_step', qa_loss.detach(), prog_bar=True)

        else:
            loss = self.calc_loss(outputs, targets, batch['pos_weight'])

        self.log('train_loss_step', loss.detach(), prog_bar=True)

        if self.args.sal_num_classes > 2:
            acc = self.calc_acc(outputs, targets)
            self.log('train_acc_step', acc, prog_bar=True)

        train_results = {'loss': loss}
        train_results['outputs'] = outputs

        if self.args.saliency_mode != 'hybrid':
            train_results['targets'] = targets
        else:
            train_results['coarse_sal_targets'] = coarse_sal_targets
            train_results['fine_sal_targets'] = fine_sal_targets

        if self.args.saliency_mode == 'hybrid':
            train_results['coarse_qa_outputs'] = coarse_qa_outputs.detach()
            train_results['fine_qa_outputs'] = fine_qa_outputs.detach()
            train_results['qa_targets'] = qa_targets.detach()
            train_results['qa_loss'] = qa_loss.detach()
            train_results['fine_qa_loss'] = fine_qa_loss.detach()
            train_results['coarse_sal_loss'] = coarse_sal_loss.detach()
            train_results['fine_sal_loss'] = fine_sal_loss.detach()
        elif self.args.pruned_qa:
            train_results['qa_outputs'] = qa_outputs.detach()
            train_results['qa_targets'] = qa_targets.detach()
            train_results['qa_loss'] = qa_loss.detach()
            train_results['sal_loss'] = sal_loss.detach()

        return train_results

    def training_epoch_end(self, results):
        self.loss_weight.step()

        if self.args.saliency_mode == 'coarse' and self.args.coarse_model == 'graph_emb':
            outputs = torch.cat([x['outputs']['sal'] for x in results])
        elif self.args.saliency_mode != 'hybrid':
            outputs = torch.cat([x['outputs'] for x in results])

        if self.args.saliency_mode != 'hybrid':
            targets = torch.cat([x['targets'] for x in results])
        else:
            coarse_sal_targets = torch.cat([x['coarse_sal_targets'] for x in results])
            fine_sal_targets = torch.cat([x['fine_sal_targets'] for x in results])

        loss = torch.stack([x['loss'] for x in results]).mean()

        self.log('train_loss_epoch', loss.detach(), prog_bar=True)
        self.log('loss_weight', self.loss_weight.data)

        if self.args.saliency_mode == 'hybrid':
            qa_loss = torch.stack([x['qa_loss'] for x in results]).mean()
            coarse_sal_loss = torch.stack([x['coarse_sal_loss'] for x in results]).mean()
            fine_qa_loss = torch.stack([x['fine_qa_loss'] for x in results]).mean()
            fine_sal_loss = torch.stack([x['fine_sal_loss'] for x in results]).mean()

            self.log('train_qa_loss_epoch', qa_loss.detach(), prog_bar=True)
            self.log('train_coarse_sal_loss_epoch', coarse_sal_loss.detach(), prog_bar=True)
            self.log('train_fine_qa_loss_epoch', fine_qa_loss.detach(), prog_bar=True)
            self.log('train_fine_sal_loss_epoch', fine_sal_loss.detach(), prog_bar=True)

            coarse_qa_outputs = torch.cat([x['coarse_qa_outputs'] for x in results])
            fine_qa_outputs = torch.cat([x['fine_qa_outputs'] for x in results])
            qa_targets = torch.cat([x['qa_targets'] for x in results])
            qa_acc = self.calc_acc(coarse_qa_outputs, qa_targets)
            fine_qa_acc = self.calc_acc(fine_qa_outputs, qa_targets)

            self.log('train_qa_acc_epoch', qa_acc)
            self.log('train_fine_qa_acc_epoch', fine_qa_acc)

        elif self.args.pruned_qa:
            qa_loss = torch.stack([x['qa_loss'] for x in results]).mean()
            sal_loss = torch.stack([x['sal_loss'] for x in results]).mean()
            self.log('train_qa_loss_epoch', qa_loss.detach(), prog_bar=True)
            self.log('train_sal_loss_epoch', sal_loss.detach(), prog_bar=True)

            qa_outputs = torch.cat([x['qa_outputs'] for x in results])
            qa_targets = torch.cat([x['qa_targets'] for x in results])
            qa_acc = self.calc_acc(qa_outputs, qa_targets)
            self.log('train_qa_acc_epoch', qa_acc)

        if self.args.saliency_mode != 'hybrid':
            if self.args.criterion in ['ce_loss', 'bce_loss', 'KL_loss'] and self.args.sal_num_classes == 2:
                f1 = self.calc_f1(outputs, targets, 'train')
                self.log('train_f1_epoch', f1['f1'], prog_bar=True)
                if self.args.criterion in ['bce_loss']:
                    self.log('train_f1_thresh_epoch', f1['f1_thresh'])

            if self.args.sal_num_classes > 2:
                acc = self.calc_acc(outputs, targets)
                self.log('train_acc_epoch', acc)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        sal_loss_weight = self.loss_weight.data

        if self.args.sal_cls_bias == 'ba_set':
            self.train_bias = self.saliency_predictor[-1].bias.data.detach().clone()
            self.saliency_predictor[-1].bias.data = self.train_bias + self.log_prior['valid'] - self.log_prior['train']
        elif self.args.sal_cls_bias == 'ba_fix':
            self.saliency_predictor[-1].bias.data = self.log_prior['valid']

        outputs = self(batch)
        if self.args.saliency_mode != 'hybrid':
            targets = batch['target']
        else:
            coarse_sal_targets = batch['coarse_sal_target']
            fine_sal_targets = batch['fine_sal_target']
        qa_targets = batch['qa_target']

        if self.args.pruned_qa:
            if self.args.saliency_mode == 'coarse':
                if self.args.coarse_model == 'ensemble':
                    sal_outputs = outputs
                    sal_probs = F.softmax(sal_outputs.reshape(batch['qa_prob'].shape), dim=-1)
                    qa_outputs = torch.sum(sal_probs * batch['qa_prob'], dim=-1)
                elif self.args.coarse_model == 'graph_emb':
                    qa_outputs = outputs['qa']
                    sal_outputs = outputs['sal']

                qa_loss = F.cross_entropy(qa_outputs, qa_targets)
                sal_loss = self.calc_loss(sal_outputs, targets, batch['pos_weight'])

                loss = qa_loss + sal_loss_weight * sal_loss
                qa_acc = self.calc_acc(qa_outputs, qa_targets)

            elif self.args.saliency_mode == 'fine':
                qa_outputs, attn_weights, attn_weights_flat, mask = outputs
                graph_sizes = batch['adj_len'] if self.args.graph_encoder == 'mhgrn' else batch['num_tuples']
                if self.args.criterion in ['bce_loss', 'mse_loss']:
                    attn_weights = attn_weights_flat

                if self.args.saliency_heuristic == "raw":
                    # use flatten attn to approximate flatten raw saliency
                    sal_loss = F.kl_div(F.log_softmax(attn_weights_flat.double()), F.softmax(batch["target_flat"]))
                else:
                    sal_loss = self.calc_loss(attn_weights, targets, batch['pos_weight'], mask, graph_sizes)
                qa_loss = F.cross_entropy(qa_outputs, qa_targets)

                loss = qa_loss + sal_loss_weight * sal_loss

                qa_acc = self.calc_acc(qa_outputs, qa_targets)

                # consistent with others config's f1
                outputs = attn_weights_flat
                if self.args.attn_head_agg == "avg_loss":
                    outputs = outputs.mean(dim=1)
                targets = batch['target'] if self.args.criterion in ['bce_loss', 'mse_loss'] else batch['target_flat']

            elif self.args.saliency_mode == 'hybrid':
                coarse_sal_outputs, fine_qa_outputs, attn_weights, attn_weights_flat, mask = outputs

                # coarse
                if self.args.coarse_model == 'ensemble':
                    fine_qa_probs = F.softmax(fine_qa_outputs.reshape(batch['qa_prob'].shape), dim=-1)
                    qa_probs = torch.stack((batch['qa_prob'], fine_qa_probs), dim=-1)
                    coarse_sal_probs = F.softmax(coarse_sal_outputs.reshape(qa_probs.shape), dim=-1)
                    coarse_qa_outputs = torch.sum(coarse_sal_probs * qa_probs, dim=-1)
                else:
                    raise NotImplementedError

                qa_loss = F.cross_entropy(coarse_qa_outputs, qa_targets)
                coarse_sal_loss = self.calc_loss(coarse_sal_outputs, coarse_sal_targets, batch['coarse_pos_weight'],
                                                 criterion='ce_loss')
                coarse_loss = qa_loss + sal_loss_weight * coarse_sal_loss

                # fine
                graph_sizes = batch['adj_len'] if self.args.graph_encoder == 'mhgrn' else batch['num_tuples']
                if self.args.criterion == 'bce_loss':
                    attn_weights = attn_weights_flat

                fine_qa_outputs = fine_qa_outputs.reshape(coarse_qa_outputs.shape)
                fine_qa_loss = F.cross_entropy(fine_qa_outputs, qa_targets)
                fine_sal_loss = self.calc_loss(attn_weights, fine_sal_targets, batch['fine_pos_weight'], mask,
                                               graph_sizes)
                fine_loss = fine_qa_loss + sal_loss_weight * fine_sal_loss
                fine_qa_acc = self.calc_acc(fine_qa_outputs, qa_targets)

                # hybrid
                loss = coarse_loss + self.args.fine_loss_weight * fine_loss
                qa_acc = self.calc_acc(coarse_qa_outputs, qa_targets)

                self.log('valid_coarse_sal_loss_step', coarse_sal_loss.detach(), prog_bar=True)

                self.log('valid_fine_qa_acc_step', fine_qa_acc, prog_bar=True)
                self.log('valid_fine_qa_loss_step', fine_qa_loss.detach(), prog_bar=True)
                self.log('valid_fine_sal_loss_step', fine_sal_loss.detach(), prog_bar=True)

            self.log('valid_qa_acc_step', qa_acc, prog_bar=True)
            self.log('valid_qa_loss_step', qa_loss.detach(), prog_bar=True)

        else:
            loss = self.calc_loss(outputs, targets, batch['pos_weight'])

        self.log('valid_loss_step', loss.detach(), prog_bar=True)

        if self.args.sal_num_classes > 2:
            acc = self.calc_acc(outputs, targets)
            self.log('valid_acc_step', acc, prog_bar=True)

        valid_results = {'loss': loss}
        valid_results['outputs'] = outputs

        if self.args.saliency_mode != 'hybrid':
            valid_results['targets'] = targets
        else:
            valid_results['coarse_sal_targets'] = coarse_sal_targets
            valid_results['fine_sal_targets'] = fine_sal_targets

        if self.args.saliency_mode == 'hybrid':
            valid_results['coarse_qa_outputs'] = coarse_qa_outputs.detach()
            valid_results['fine_qa_outputs'] = fine_qa_outputs.detach()
            valid_results['qa_targets'] = qa_targets.detach()
            valid_results['qa_loss'] = qa_loss.detach()
            valid_results['fine_qa_loss'] = fine_qa_loss.detach()
            valid_results['coarse_sal_loss'] = coarse_sal_loss.detach()
            valid_results['fine_sal_loss'] = fine_sal_loss.detach()
        elif self.args.pruned_qa:
            valid_results['qa_outputs'] = qa_outputs.detach()
            valid_results['qa_targets'] = qa_targets.detach()
            valid_results['qa_loss'] = qa_loss.detach()
            valid_results['sal_loss'] = sal_loss.detach()

        return valid_results

    def validation_epoch_end(self, results):
        if self.args.saliency_mode == 'coarse' and self.args.coarse_model == 'graph_emb':
            outputs = torch.cat([x['outputs']['sal'] for x in results])
        elif self.args.saliency_mode != 'hybrid':
            outputs = torch.cat([x['outputs'] for x in results])

        if self.args.saliency_mode != 'hybrid':
            targets = torch.cat([x['targets'] for x in results])
        else:
            coarse_sal_targets = torch.cat([x['coarse_sal_targets'] for x in results])
            fine_sal_targets = torch.cat([x['fine_sal_targets'] for x in results])

        loss = torch.stack([x['loss'] for x in results]).mean()

        self.log('valid_loss_epoch', loss.detach(), prog_bar=True)

        if self.args.saliency_mode == 'hybrid':
            qa_loss = torch.stack([x['qa_loss'] for x in results]).mean()
            coarse_sal_loss = torch.stack([x['coarse_sal_loss'] for x in results]).mean()
            fine_qa_loss = torch.stack([x['fine_qa_loss'] for x in results]).mean()
            fine_sal_loss = torch.stack([x['fine_sal_loss'] for x in results]).mean()

            self.log('valid_qa_loss_epoch', qa_loss.detach(), prog_bar=True)
            self.log('valid_coarse_sal_loss_epoch', coarse_sal_loss.detach(), prog_bar=True)
            self.log('valid_fine_qa_loss_epoch', fine_qa_loss.detach(), prog_bar=True)
            self.log('valid_fine_sal_loss_epoch', fine_sal_loss.detach(), prog_bar=True)

            coarse_qa_outputs = torch.cat([x['coarse_qa_outputs'] for x in results])
            fine_qa_outputs = torch.cat([x['fine_qa_outputs'] for x in results])
            qa_targets = torch.cat([x['qa_targets'] for x in results])
            qa_acc = self.calc_acc(coarse_qa_outputs, qa_targets)
            fine_qa_acc = self.calc_acc(fine_qa_outputs, qa_targets)

            self.log('valid_qa_acc_epoch', qa_acc)
            self.log('valid_fine_qa_acc_epoch', fine_qa_acc)

        elif self.args.pruned_qa:
            qa_loss = torch.stack([x['qa_loss'] for x in results]).mean()
            sal_loss = torch.stack([x['sal_loss'] for x in results]).mean()
            self.log('valid_qa_loss_epoch', qa_loss.detach(), prog_bar=True)
            self.log('valid_sal_loss_epoch', sal_loss.detach(), prog_bar=True)

            qa_outputs = torch.cat([x['qa_outputs'] for x in results])
            qa_targets = torch.cat([x['qa_targets'] for x in results])
            qa_acc = self.calc_acc(qa_outputs, qa_targets)
            self.log('valid_qa_acc_epoch', qa_acc)

        if self.args.saliency_mode != 'hybrid':
            if self.args.criterion in ['ce_loss', 'bce_loss', 'KL_loss'] and self.args.sal_num_classes == 2:
                f1 = self.calc_f1(outputs, targets, 'valid')
                self.log('valid_f1_epoch', f1['f1'], prog_bar=True)
                if self.args.criterion in ['bce_loss']:
                    self.log('valid_f1_thresh_epoch', f1['f1_thresh'])

            if self.args.sal_num_classes > 2:
                acc = self.calc_acc(outputs, targets)
                self.log('valid_acc_epoch', acc)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        sal_loss_weight = self.loss_weight.data
        if self.args.save_saliency and self.args.criterion in ['bce_loss'] and self.args.saliency_mode == 'coarse':
            print('Our calc_f1 for bce_loss is sensitive to split, fix this first.')
            raise NotImplementedError

        if self.args.sal_cls_bias == 'ba_set':
            self.train_bias = self.saliency_predictor[-1].bias.data.detach().clone()
            self.saliency_predictor[-1].bias.data = self.train_bias + self.log_prior['valid'] - self.log_prior['train']
        elif self.args.sal_cls_bias == 'ba_fix':
            self.saliency_predictor[-1].bias.data = self.log_prior['valid']

        outputs = self(batch)
        if self.args.saliency_mode != 'hybrid':
            targets = batch['target']
        else:
            coarse_sal_targets = batch['coarse_sal_target']
            fine_sal_targets = batch['fine_sal_target']
        qa_targets = batch['qa_target']

        if self.args.pruned_qa:
            if self.args.saliency_mode == 'coarse':
                if self.args.coarse_model == 'ensemble':
                    sal_outputs = outputs
                    sal_probs = F.softmax(sal_outputs.reshape(batch['qa_prob'].shape), dim=-1)
                    qa_outputs = torch.sum(sal_probs * batch['qa_prob'], dim=-1)
                elif self.args.coarse_model == 'graph_emb':
                    qa_outputs = outputs['qa']
                    sal_outputs = outputs['sal']

                qa_loss = F.cross_entropy(qa_outputs, qa_targets)
                sal_loss = self.calc_loss(sal_outputs, targets, batch['pos_weight'])

                loss = qa_loss + sal_loss_weight * sal_loss
                qa_acc = self.calc_acc(qa_outputs, qa_targets)

            elif self.args.saliency_mode == 'fine':
                qa_outputs, attn_weights, attn_weights_flat, mask = outputs
                graph_sizes = batch['adj_len'] if self.args.graph_encoder == 'mhgrn' else batch['num_tuples']
                if self.args.criterion in ['bce_loss', 'mse_loss']:
                    attn_weights = attn_weights_flat

                if self.args.saliency_heuristic == "raw":
                    # use flatten attn to approximate flatten raw saliency
                    sal_loss = F.kl_div(F.log_softmax(attn_weights_flat.double()), F.softmax(batch["target_flat"]))
                else:
                    sal_loss = self.calc_loss(attn_weights, targets, batch['pos_weight'], mask, graph_sizes)
                qa_loss = F.cross_entropy(qa_outputs, qa_targets)

                loss = qa_loss + sal_loss_weight * sal_loss

                qa_acc = self.calc_acc(qa_outputs, qa_targets)

                # consistent with others config's f1
                outputs = attn_weights_flat
                if self.args.attn_head_agg == "avg_loss":
                    outputs = outputs.mean(dim=1)
                targets = batch['target'] if self.args.criterion in ['bce_loss', 'mse_loss'] else batch['target_flat']

            elif self.args.saliency_mode == 'hybrid':
                coarse_sal_outputs, fine_qa_outputs, attn_weights, attn_weights_flat, mask = outputs

                # coarse
                if self.args.coarse_model == 'ensemble':
                    fine_qa_probs = F.softmax(fine_qa_outputs.reshape(batch['qa_prob'].shape), dim=-1)
                    qa_probs = torch.stack((batch['qa_prob'], fine_qa_probs), dim=-1)
                    coarse_sal_probs = F.softmax(coarse_sal_outputs.reshape(qa_probs.shape), dim=-1)
                    coarse_qa_outputs = torch.sum(coarse_sal_probs * qa_probs, dim=-1)
                else:
                    raise NotImplementedError

                qa_loss = F.cross_entropy(coarse_qa_outputs, qa_targets)
                coarse_sal_loss = self.calc_loss(coarse_sal_outputs, coarse_sal_targets, batch['coarse_pos_weight'],
                                                 criterion='ce_loss')
                coarse_loss = qa_loss + sal_loss_weight * coarse_sal_loss

                # fine
                graph_sizes = batch['adj_len'] if self.args.graph_encoder == 'mhgrn' else batch['num_tuples']
                if self.args.criterion == 'bce_loss':
                    attn_weights = attn_weights_flat

                fine_qa_outputs = fine_qa_outputs.reshape(coarse_qa_outputs.shape)
                fine_qa_loss = F.cross_entropy(fine_qa_outputs, qa_targets)
                fine_sal_loss = self.calc_loss(attn_weights, fine_sal_targets, batch['fine_pos_weight'], mask,
                                               graph_sizes)
                fine_loss = fine_qa_loss + sal_loss_weight * fine_sal_loss
                fine_qa_acc = self.calc_acc(fine_qa_outputs, qa_targets)

                # hybrid
                loss = coarse_loss + self.args.fine_loss_weight * fine_loss
                qa_acc = self.calc_acc(coarse_qa_outputs, qa_targets)

                self.log('test_coarse_sal_loss_step', coarse_sal_loss.detach(), prog_bar=True)

                self.log('test_fine_qa_acc_step', fine_qa_acc, prog_bar=True)
                self.log('test_fine_qa_loss_step', fine_qa_loss.detach(), prog_bar=True)
                self.log('test_fine_sal_loss_step', fine_sal_loss.detach(), prog_bar=True)

            self.log('test_qa_acc_step', qa_acc, prog_bar=True)
            self.log('test_qa_loss_step', qa_loss.detach(), prog_bar=True)

        else:
            loss = self.calc_loss(outputs, targets, batch['pos_weight'])

        self.log('test_loss_step', loss.detach(), prog_bar=True)

        if self.args.sal_num_classes > 2:
            acc = self.calc_acc(outputs, targets)
            self.log('test_acc_step', acc, prog_bar=True)

        test_results = {'loss': loss}
        test_results['outputs'] = outputs

        if self.args.saliency_mode != 'hybrid':
            test_results['targets'] = targets
        else:
            test_results['coarse_sal_targets'] = coarse_sal_targets
            test_results['fine_sal_targets'] = fine_sal_targets

        if self.args.saliency_mode == 'hybrid':
            test_results['coarse_qa_outputs'] = coarse_qa_outputs.detach()
            test_results['fine_qa_outputs'] = fine_qa_outputs.detach()
            test_results['qa_targets'] = qa_targets.detach()
            test_results['qa_loss'] = qa_loss.detach()
            test_results['fine_qa_loss'] = fine_qa_loss.detach()
            test_results['coarse_sal_loss'] = coarse_sal_loss.detach()
            test_results['fine_sal_loss'] = fine_sal_loss.detach()
        elif self.args.pruned_qa:
            test_results['qa_outputs'] = qa_outputs.detach()
            test_results['qa_targets'] = qa_targets.detach()
            test_results['qa_loss'] = qa_loss.detach()
            test_results['sal_loss'] = sal_loss.detach()

        return test_results

    def test_epoch_end(self, results):
        if self.args.save_saliency and self.args.criterion in ['bce_loss'] and self.args.saliency_mode == 'coarse':
            print('Our calc_f1 for bce_loss is sensitive to split, fix this first.')
            raise NotImplementedError

        if self.args.saliency_mode == 'coarse' and self.args.coarse_model == 'graph_emb':
            outputs = torch.cat([x['outputs']['sal'] for x in results])
        elif self.args.saliency_mode != 'hybrid':
            outputs = torch.cat([x['outputs'] for x in results])

        if self.args.saliency_mode != 'hybrid':
            targets = torch.cat([x['targets'] for x in results])
        else:
            coarse_sal_targets = torch.cat([x['coarse_sal_targets'] for x in results])
            fine_sal_targets = torch.cat([x['fine_sal_targets'] for x in results])

        loss = torch.stack([x['loss'] for x in results]).mean()

        self.log('test_loss_epoch', loss.detach(), prog_bar=True)

        if self.args.saliency_mode == 'hybrid':
            qa_loss = torch.stack([x['qa_loss'] for x in results]).mean()
            coarse_sal_loss = torch.stack([x['coarse_sal_loss'] for x in results]).mean()
            fine_qa_loss = torch.stack([x['fine_qa_loss'] for x in results]).mean()
            fine_sal_loss = torch.stack([x['fine_sal_loss'] for x in results]).mean()

            self.log('test_qa_loss_epoch', qa_loss.detach(), prog_bar=True)
            self.log('test_coarse_sal_loss_epoch', coarse_sal_loss.detach(), prog_bar=True)
            self.log('test_fine_qa_loss_epoch', fine_qa_loss.detach(), prog_bar=True)
            self.log('test_fine_sal_loss_epoch', fine_sal_loss.detach(), prog_bar=True)

            coarse_qa_outputs = torch.cat([x['coarse_qa_outputs'] for x in results])
            fine_qa_outputs = torch.cat([x['fine_qa_outputs'] for x in results])
            qa_targets = torch.cat([x['qa_targets'] for x in results])
            qa_acc = self.calc_acc(coarse_qa_outputs, qa_targets)
            fine_qa_acc = self.calc_acc(fine_qa_outputs, qa_targets)

            self.log('test_qa_acc_epoch', qa_acc)
            self.log('test_fine_qa_acc_epoch', fine_qa_acc)

        elif self.args.pruned_qa:
            qa_loss = torch.stack([x['qa_loss'] for x in results]).mean()
            sal_loss = torch.stack([x['sal_loss'] for x in results]).mean()
            self.log('test_qa_loss_epoch', qa_loss.detach(), prog_bar=True)
            self.log('test_sal_loss_epoch', sal_loss.detach(), prog_bar=True)

            qa_outputs = torch.cat([x['qa_outputs'] for x in results])
            qa_targets = torch.cat([x['qa_targets'] for x in results])
            qa_acc = self.calc_acc(qa_outputs, qa_targets)
            self.log('test_qa_acc_epoch', qa_acc)

        if self.args.saliency_mode != 'hybrid':
            if self.args.criterion in ['ce_loss', 'bce_loss', 'KL_loss'] and self.args.sal_num_classes == 2:
                f1 = self.calc_f1(outputs, targets, 'test')
                self.log('test_f1_epoch', f1['f1'], prog_bar=True)
                if self.args.criterion in ['bce_loss']:
                    self.log('test_f1_thresh_epoch', f1['f1_thresh'])

            if self.args.sal_num_classes > 2:
                acc = self.calc_acc(outputs, targets)
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

        if self.args.saliency_mode == 'hybrid':
            optimizer_grouped_parameters += [
                {
                    'params': [p for n, p in self.coarse_saliency_gnn.named_parameters() if
                               not any(nd in n for nd in no_decay)],
                    'weight_decay': self.args.weight_decay,
                    'lr': self.args.graph_lr
                },
                {
                    'params': [p for n, p in self.coarse_saliency_gnn.named_parameters() if
                               any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': self.args.graph_lr
                },
                {
                    'params': [p for n, p in self.fine_saliency_gnn.named_parameters() if
                               not any(nd in n for nd in no_decay)],
                    'weight_decay': self.args.weight_decay,
                    'lr': self.args.fine_graph_lr
                },
                {
                    'params': [p for n, p in self.fine_saliency_gnn.named_parameters() if
                               any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': self.args.fine_graph_lr
                }
            ]
        else:
            optimizer_grouped_parameters += [
                {
                    'params': [p for n, p in self.saliency_gnn.named_parameters() if
                               not any(nd in n for nd in no_decay)],
                    'weight_decay': self.args.weight_decay,
                    'lr': self.args.graph_lr
                },
                {
                    'params': [p for n, p in self.saliency_gnn.named_parameters() if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': self.args.graph_lr
                },
            ]

        if not (self.args.pruned_qa and self.args.saliency_mode == "fine"):
            optimizer_grouped_parameters += [
                {
                    'params': [p for n, p in self.saliency_predictor.named_parameters() if
                               not any(nd in n for nd in no_decay)],
                    'weight_decay': self.args.weight_decay,
                    'lr': self.args.graph_lr
                },
                {
                    'params': [p for n, p in self.saliency_predictor.named_parameters() if
                               any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': self.args.graph_lr
                }
            ]

        if self.args.sal_att_pooling:
            att_params = [
                {
                    'params': [p for n, p in self.graph2att.named_parameters() if not any(nd in n for nd in no_decay)],
                    'weight_decay': self.args.weight_decay,
                    'lr': self.args.graph_lr
                },
                {
                    'params': [p for n, p in self.text2att.named_parameters() if not any(nd in n for nd in no_decay)],
                    'weight_decay': self.args.weight_decay,
                    'lr': self.args.graph_lr
                },
                {
                    'params': [p for n, p in self.attention.named_parameters() if not any(nd in n for nd in no_decay)],
                    'weight_decay': self.args.weight_decay,
                    'lr': self.args.graph_lr
                },
            ]

            optimizer_grouped_parameters += att_params

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
        parser.add_argument('--text_lr', default=2e-5, type=float)
        parser.add_argument('--graph_lr', default=1e-3, type=float)
        parser.add_argument('--fine_graph_lr', default=1e-3, type=float)
        parser.add_argument('--adam_epsilon', default=1e-8, type=float)
        parser.add_argument('--warmup_updates', default=0, type=int)
        parser.add_argument('--weight_decay', default=0.0, type=float)
        parser.add_argument('--lr_scheduler', default='fixed', type=str, choices=['fixed', 'linear_with_warmup'])
        parser.add_argument('--arch', default='albert_xxlarge_v2', type=str,
                            choices=['albert_xxlarge_v2', 'roberta_base', 'roberta_large'])
        parser.add_argument('--criterion', default='ce_loss', type=str,
                            choices=['bce_loss', 'ce_loss', 'mse_loss', 'mae_loss', 'coral_loss', 'KL_loss'])
        parser.add_argument('--attn_bound', default='100', type=int)
        # if avg_heads, loss(avg of attn scores , target)
        # if avg_loss, avg_h(loss(attn scores of head h, target))
        parser.add_argument('--attn_head_agg', default='avg_head', type=str, choices=['avg_head', 'avg_loss'])
        # only valid if attn_head_agg = avg_loss
        # format <mode>-<k>
        # mode in {random, first, random_step}
        # random: before training select k index from [0, 1, ... graph_att_head_num-1]
        # first: select [0, 1, ..., k-1]
        # random_step: at each step select k index from [0, 1, ... graph_att_head_num-1]
        # k <= graph_att_head_num
        parser.add_argument('--attn_agg_k', default="none", type=str)

        parser.add_argument('--graph_encoder', default='mhgrn', type=str, choices=['rn', 'mhgrn'])
        parser.add_argument('--sal_gnn_layer_type', default='mhgrn', type=str, choices=['mhgrn', 'rn', 'pathgen'])
        parser.add_argument('--sal_gnn_io_dim', default=256, type=int)
        parser.add_argument('--sal_gnn_hidden_dim', default=256, type=int)
        parser.add_argument('--sal_gnn_hidden_layers', default=1, type=int)
        parser.add_argument('--sal_gnn_layer_norm', default=True, type=bool)
        parser.add_argument('--sal_gnn_dropout', default=False, type=bool)
        parser.add_argument('--sal_gnn_layers', default=1, type=int)
        parser.add_argument('--sal_cls_hidden_dim', default=1024, type=int)
        parser.add_argument('--sal_cls_hidden_layers', default=1, type=int)
        parser.add_argument('--sal_cls_layer_norm', default=True, type=bool)
        parser.add_argument('--sal_cls_dropout', default=False, type=bool)
        parser.add_argument('--sal_cls_bias', default='none', type=str, choices=['none', 'ba_set', 'ba_fix'])
        parser.add_argument('--sal_num_classes', default=2, type=int)
        parser.add_argument('--sal_loss_weight', default=1, type=float)
        parser.add_argument('--fine_sal_loss_weight', default=1, type=float)
        parser.add_argument('--fine_loss_weight', default=1, type=float)
        parser.add_argument('--coarse_model', default='ensemble', type=str, choices=['ensemble', 'graph_emb'])
        parser.add_argument('--no_kg_emb', default='learned', type=str, choices=['learned', 'random', 'zero'])

        parser.add_argument('--graph_init_range', default=0.02, type=float)
        parser.add_argument('--graph_init_rn', action='store_true')
        parser.add_argument('--graph_init_identity', action='store_true')
        parser.add_argument('--graph_pool', default='multihead_pool', type=str,
                            choices=['none', 'multihead_pool', 'att_pool'])
        parser.add_argument('--graph_mlp_dim', default=128, type=int)
        parser.add_argument('--graph_k', default=2, type=int)
        parser.add_argument('--graph_n_type', default=3, type=int)
        parser.add_argument('--graph_num_relation', default=34, type=int)
        parser.add_argument('--graph_num_basis', default=0, type=int)
        parser.add_argument('--graph_gnn_layer_num', default=1, type=int)
        parser.add_argument('--graph_diag_decompose', action='store_true')
        parser.add_argument('--graph_att_head_num', default=2, type=int)
        parser.add_argument('--graph_att_dim', default=50, type=int)
        parser.add_argument('--graph_mlp_layer_num', default=2, type=int)
        parser.add_argument('--graph_dropoutm', default=0.3, type=float)
        parser.add_argument('--graph_att_layer_num', default=1, type=int)
        parser.add_argument('--graph_dropouti', default=0.1, type=float)
        parser.add_argument('--graph_dropoutg', default=0.2, type=float)
        parser.add_argument('--graph_dropoutf', default=0.2, type=float)
        parser.add_argument('--graph_eps', default=1e-15, type=float)
        parser.add_argument('--graph_fc_dim', default=200, type=int)
        parser.add_argument('--graph_fc_layer_num', default=0, type=int)
        parser.add_argument('--graph_freeze_ent_emb', action='store_true')

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

        parser.add_argument('--sal_att_pooling', action='store_true')
        parser.add_argument('--sal_att_input_dim', default=512, type=int)
        parser.add_argument('--sal_num_att_head', default=2, type=int)
        parser.add_argument('--input_dim_gpt', default=768, type=int)

        parser.add_argument('--fine_checkpoint_path', default=None, type=str)

        return parser
