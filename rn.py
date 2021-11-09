import numpy as np
import torch
from torch import nn

from layers import MLP, AttPoolLayer, MultiheadAttPoolLayer


class RelationNetModel(nn.Module):
    def __init__(self, args, concept_emb, rel_emb, *unused):
        super().__init__()
        self.args = args
        self.ablation = []

        self.concept_emb = concept_emb
        concept_dim = self.concept_emb.emb.embedding_dim

        self.rel_emb = rel_emb
        self.relation_num = rel_emb.num_embeddings
        relation_dim = rel_emb.embedding_dim

        hidden_size = args.graph_mlp_dim
        num_hidden_layers = args.graph_mlp_layer_num
        dropout = args.graph_dropoutm
        sent_dim = args.text_out_dim
        fc_size = args.graph_fc_dim
        num_fc_layers = args.graph_fc_layer_num
        num_attention_heads = args.graph_att_head_num
        self.emb_data = None
        self.pool = args.graph_pool

        self.init_range = args.graph_init_range
        self.do_init_rn = args.graph_init_rn
        self.do_init_identity = args.graph_init_identity

        encoder_dim = {'no_qa': relation_dim, 'no_2hop_qa': relation_dim, 'no_rel': concept_dim * 2}.get(None,
                                                                                                         concept_dim * 2 + relation_dim)
        self.mlp = MLP(encoder_dim, hidden_size * 2, hidden_size,
                       num_hidden_layers, dropout, batch_norm=False, layer_norm=True)

        if self.pool == 'multihead_pool':
            self.attention = MultiheadAttPoolLayer(num_attention_heads, sent_dim, hidden_size)
        elif self.pool == 'att_pool':
            self.attention = AttPoolLayer(sent_dim, hidden_size)

        self.dropout_m = nn.Dropout(dropout)
        if args.graph_encoder == 'pathgen':
            self.hid2out = MLP(args.input_dim_gpt + hidden_size + sent_dim, fc_size, 1, num_fc_layers, dropout,
                               batch_norm=False, layer_norm=True)
        else:
            self.hid2out = MLP(hidden_size + sent_dim, fc_size, 1, num_fc_layers, dropout, batch_norm=False,
                               layer_norm=True)
        self.activation = nn.GELU()

        if self.init_range > 0:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, batch, sent_vecs, path_embedding=None, bypass_logits=False, end2end=False, return_raw_attn=False,
                goccl=False):
        """
        sent_vecs: tensor of shape (batch_size, d_sent)
        qa_ids: tensor of shape (batch_size, max_tuple_num, 2)
        rel_ids: tensor of shape (batch_size, max_tuple_num)
        num_tuples: tensor of shape (batch_size,)
        (emb_data: tensor of shape (batch_size, max_cpt_num, emb_dim))
        """

        qa_ids = batch['qa_ids']
        rel_ids = batch['rel_ids']
        num_tuples = batch['num_tuples']

        bs, sl, _ = qa_ids.size()
        mask = torch.arange(sl, device=qa_ids.device) >= num_tuples.unsqueeze(1)
        if self.args.saliency_mode == 'fine' and self.args.saliency_source == 'target' and self.args.task == 'qa' and (
                self.args.save_saliency == False or self.args.save_salkg_fine_target_preds):
            assert mask.shape == batch['saliency_results'].shape
            mask = mask | ~batch['saliency_results']
        mask[mask.all(1), 0] = 0  # a temporary solution for instances that have no qar-pairs

        qa_emb = self.concept_emb(qa_ids.view(bs, -1), self.emb_data).view(bs, sl, -1)
        rel_embed = self.rel_emb(rel_ids)

        if self.args.save_saliency and self.args.saliency_mode == 'fine' and self.args.saliency_method == 'grad':
            qa_emb.requires_grad = True

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

        concat = torch.cat((qa_emb, rel_embed), -1)

        qars_vecs = self.mlp(concat)
        qars_vecs = self.activation(qars_vecs)

        if self.pool in ['multihead_pool', 'att_pool']:
            pooled_vecs, att_scores = self.attention(sent_vecs, qars_vecs, mask, return_raw_attn)
        else:
            qars_vecs = qars_vecs.masked_fill(mask.unsqueeze(2).expand_as(qars_vecs), 0)
            pooled_vecs = qars_vecs.sum(1) / (~mask).sum(1).unsqueeze(1).to(qars_vecs.device)
            att_scores = None

        if bypass_logits:
            mask = torch.arange(sl, device=qa_ids.device) < num_tuples.unsqueeze(1)
            return qars_vecs, mask, pooled_vecs, att_scores

        if path_embedding != None:
            logits = self.hid2out(self.dropout_m(torch.cat((path_embedding, pooled_vecs, sent_vecs), 1)))
            if goccl:
                baseline_vecs = torch.zeros_like(pooled_vecs).to(pooled_vecs.device)
                baseline_logits = self.hid2out(self.dropout_m(torch.cat((path_embedding, baseline_vecs, sent_vecs), 1)))
                return logits, baseline_logits
        else:
            logits = self.hid2out(self.dropout_m(torch.cat((pooled_vecs, sent_vecs), 1)))
            if goccl:
                baseline_vecs = torch.zeros_like(pooled_vecs).to(pooled_vecs.device)
                baseline_logits = self.hid2out(self.dropout_m(torch.cat((baseline_vecs, sent_vecs), 1)))
                return logits, baseline_logits

        if end2end:
            mask = torch.arange(sl, device=qa_ids.device) < num_tuples.unsqueeze(1)
            # node emb, mask, graph emb
            return qars_vecs, mask, pooled_vecs, att_scores, logits

        if self.args.save_saliency and self.args.saliency_method == 'grad':
            if self.args.saliency_mode == 'coarse':
                sal_units = pooled_vecs
            elif self.args.saliency_mode == 'fine':
                sal_units = (qa_emb, rel_embed)
            return logits, att_scores, sal_units

        return logits, att_scores, None
