# from data_utils import *

import numpy as np
import torch
from torch import nn

from layers import MLP, AttPoolLayer, CustomizedEmbedding, MultiheadAttPoolLayer
from utils import cal_2hop_rel_emb


def init_weights_normal(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.02)


class Path_Encoder(nn.Module):
    """docstring for Classifier"""

    def __init__(self, input_dim_bert, input_dim_gpt=768):
        super().__init__()

        self.input_dim_gpt = input_dim_gpt
        self.input_dim_bert = input_dim_bert

        self.attention = nn.Sequential(
            nn.Linear(self.input_dim_gpt, self.input_dim_bert),
            nn.Tanh(),
        )
        self.attention.apply(init_weights_normal)

    def forward(self, s, p):
        # choice: [batch, hidden]
        # context: [batch, context, hidden]

        batch_size, num_context, _ = p.size()

        # attention
        # q_T*W(p)
        query = s.view(batch_size, 1, self.input_dim_bert)
        alpha = (self.attention(p) * query).sum(-1, keepdim=True)
        alpha = alpha.softmax(dim=-2)
        context = (alpha * p).sum(-2)

        return context


class LMRelationNetModel(nn.Module):
    def __init__(self, args, concept_emb, rel_emb, sent_dim):
        super().__init__()
        self.args = args

        pretrained_concept_emb = torch.from_numpy(np.load(args.ent_emb_path))
        concept_num, concept_dim = pretrained_concept_emb.shape

        rel_emb_ = np.load(self.args.rel_emb_path)
        rel_emb_ = np.concatenate((rel_emb_, -rel_emb_), 0)
        rel_emb_ = cal_2hop_rel_emb(rel_emb_)
        rel_emb_ = torch.tensor(rel_emb_)
        relation_num, relation_dim = rel_emb_.shape

        self.decoder = RelationNet(args, concept_num, concept_dim, relation_num, relation_dim, sent_dim, concept_dim,
                                   args.graph_mlp_dim, args.graph_mlp_layer_num, args.graph_att_head_num,
                                   args.graph_fc_dim,
                                   args.graph_fc_layer_num, args.graph_dropoutm, pretrained_concept_emb, rel_emb_,
                                   freeze_ent_emb=args.graph_freeze_ent_emb,
                                   init_range=args.graph_init_range, ablation=args.graph_pool)
        self.path_encoder = Path_Encoder(sent_dim)

    def forward(self, batch, sent_vecs, bypass_logits=False, end2end=False, return_raw_attn=False, goccl=False):
        path_embedding = batch['path_emb']
        path_embedding = path_embedding.view(path_embedding.size(0) * path_embedding.size(1),
                                             *path_embedding.size()[2:])
        agg_path_embedding = self.path_encoder(s=sent_vecs, p=path_embedding)
        if end2end or bypass_logits:
            return self.decoder(batch, agg_path_embedding, sent_vecs, batch['qa_ids'], batch['rel_ids'],
                                batch['num_tuples'],
                                bypass_logits=bypass_logits,
                                end2end=end2end,
                                return_raw_attn=return_raw_attn
                                )
        if goccl:
            return self.decoder(batch, agg_path_embedding, sent_vecs, batch['qa_ids'], batch['rel_ids'],
                                batch['num_tuples'], goccl=goccl)
        logits, att_scores, sal_units = self.decoder(batch, agg_path_embedding, sent_vecs, batch['qa_ids'],
                                                     batch['rel_ids'], batch['num_tuples'], bypass_logits=bypass_logits)
        return logits, att_scores, sal_units


class RelationNet(nn.Module):

    def __init__(self, args, concept_num, concept_dim, relation_num, relation_dim, sent_dim, concept_in_dim,
                 hidden_size, num_hidden_layers, num_attention_heads, fc_size, num_fc_layers, dropout,
                 pretrained_concept_emb=None, pretrained_relation_emb=None, freeze_ent_emb=True,
                 init_range=0, ablation=None, use_contextualized=False, emb_scale=1.0, path_embedding_dim=768):

        super().__init__()
        self.args = args
        self.init_range = init_range
        self.relation_num = relation_num
        self.ablation = ablation

        self.rel_emb = nn.Embedding(relation_num, relation_dim)
        self.concept_emb = CustomizedEmbedding(concept_num=concept_num, concept_out_dim=concept_dim,
                                               use_contextualized=use_contextualized, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb,
                                               freeze_ent_emb=freeze_ent_emb,
                                               scale=emb_scale)

        encoder_dim = {'no_qa': relation_dim, 'no_2hop_qa': relation_dim, 'no_rel': concept_dim * 2}.get(self.ablation,
                                                                                                         concept_dim * 2 + relation_dim)
        if self.ablation in ('encode_qas',):
            encoder_dim += sent_dim
        self.mlp = MLP(encoder_dim, hidden_size * 2, hidden_size,
                       num_hidden_layers, dropout, batch_norm=False, layer_norm=True)

        if ablation in ('multihead_pool',):
            self.attention = MultiheadAttPoolLayer(num_attention_heads, sent_dim, hidden_size)
        elif ablation in ('att_pool',):
            self.attention = AttPoolLayer(sent_dim, hidden_size)

        self.dropout_m = nn.Dropout(dropout)
        self.hid2out = MLP(path_embedding_dim + hidden_size + sent_dim, fc_size, 1, num_fc_layers, dropout,
                           batch_norm=False, layer_norm=True)
        self.activation = nn.GELU()

        if self.init_range > 0:
            self.apply(self._init_weights)

        if pretrained_relation_emb is not None and ablation not in ('randomrel',):
            self.rel_emb.weight.data.copy_(pretrained_relation_emb)

        if pretrained_concept_emb is not None and not use_contextualized:
            self.concept_emb.emb.weight.data.copy_(pretrained_concept_emb)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, batch, path_embedding, sent_vecs, qa_ids, rel_ids, num_tuples, emb_data=None, bypass_logits=False,
                end2end=False, return_raw_attn=False, goccl=False):
        """
        sent_vecs: tensor of shape (batch_size, d_sent)
        qa_ids: tensor of shape (batch_size, max_tuple_num, 2)
        rel_ids: tensor of shape (batch_size, max_tuple_num)
        num_tuples: tensor of shape (batch_size,)
        (emb_data: tensor of shape (batch_size, max_cpt_num, emb_dim))
        """

        bs, sl, _ = qa_ids.size()
        mask = torch.arange(sl, device=qa_ids.device) >= num_tuples.unsqueeze(1)
        if self.ablation in ('no_1hop', 'no_2hop', 'no_2hop_qa'):
            n_1hop_rel = int(np.sqrt(self.relation_num))
            assert n_1hop_rel * (n_1hop_rel + 1) == self.relation_num
            valid_mask = rel_ids > n_1hop_rel if self.ablation == 'no_1hop' else rel_ids <= n_1hop_rel
            mask = mask | ~valid_mask
        if self.args.saliency_mode == 'fine' and self.args.saliency_source == 'target' and self.args.task == 'qa' and (
                self.args.save_saliency == False or self.args.save_salkg_fine_target_preds):
            assert mask.shape == batch['saliency_results'].shape
            mask = mask | ~batch['saliency_results']
        mask[mask.all(1), 0] = 0  # a temporary solution for instances that have no qar-pairs

        qa_emb = self.concept_emb(qa_ids.view(bs, -1), emb_data).view(bs, sl, -1)
        rel_embed = self.rel_emb(rel_ids)

        if self.args.save_saliency and self.args.saliency_mode == 'fine' and self.args.saliency_method == 'grad':
            qa_emb.requires_grad = True

        if self.ablation not in ('no_factor_mul',):
            n_1hop_rel = int(np.sqrt(self.relation_num))
            assert n_1hop_rel * (n_1hop_rel + 1) == self.relation_num
            rel_ids = rel_ids.view(bs * sl)
            twohop_mask = rel_ids >= n_1hop_rel
            twohop_rel = rel_ids[twohop_mask] - n_1hop_rel
            r1, r2 = twohop_rel // n_1hop_rel, twohop_rel % n_1hop_rel
            assert (r1 >= 0).all() and (r2 >= 0).all() and (r1 < n_1hop_rel).all() and (r2 < n_1hop_rel).all()
            rel_embed = rel_embed.view(bs * sl, -1)
            rel_embed[twohop_mask] = torch.mul(self.rel_emb(r1), self.rel_emb(r2))
            rel_embed = rel_embed.view(bs, sl, -1)

        if self.ablation in ('no_qa', 'no_rel', 'no_2hop_qa'):
            concat = rel_embed if self.ablation in ('no_qa', 'no_2hop_qa') else qa_emb
        else:
            concat = torch.cat((qa_emb, rel_embed), -1)

        if self.ablation in ('encode_qas',):
            sent_vecs_expanded = sent_vecs.unsqueeze(1).expand(bs, sl, -1)
            concat = torch.cat((concat, sent_vecs_expanded), -1)

        qars_vecs = self.mlp(concat)
        qars_vecs = self.activation(qars_vecs)

        if self.ablation in ('multihead_pool', 'att_pool'):
            pooled_vecs, att_scores = self.attention(sent_vecs, qars_vecs, mask, return_raw_attn)
        else:
            qars_vecs = qars_vecs.masked_fill(mask.unsqueeze(2).expand_as(qars_vecs), 0)
            pooled_vecs = qars_vecs.sum(1) / (~mask).float().sum(1).unsqueeze(1).float().to(qars_vecs.device)
            att_scores = None

        if self.ablation == 'no_kg':
            pooled_vecs[:] = 0

        if bypass_logits:
            mask = torch.arange(sl, device=qa_ids.device) < num_tuples.unsqueeze(1)
            return qars_vecs, mask, pooled_vecs, att_scores

        logits = self.hid2out(self.dropout_m(torch.cat((path_embedding, pooled_vecs, sent_vecs), 1)))

        if goccl:
            baseline_vecs = torch.zeros_like(pooled_vecs).to(pooled_vecs.device)
            baseline_logits = self.hid2out(self.dropout_m(torch.cat((path_embedding, baseline_vecs, sent_vecs), 1)))
            return logits, baseline_logits

        if end2end:
            mask = torch.arange(sl, device=qa_ids.device) < num_tuples.unsqueeze(1)
            return qars_vecs, mask, pooled_vecs, att_scores, logits

        if self.args.save_saliency and self.args.saliency_method == 'grad':
            if self.args.saliency_mode == 'coarse':
                sal_units = pooled_vecs
            elif self.args.saliency_mode == 'fine':
                sal_units = (qa_emb, rel_embed)
            return logits, att_scores, sal_units

        return logits, att_scores, None
