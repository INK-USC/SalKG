import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from layers import MLP, MultiheadAttPoolLayer, TypedLinear


class GraphRelationNetModel(nn.Module):
    def __init__(self, args, concept_emb, *unused):
        super().__init__()
        self.args = args
        self.ablation = []

        self.init_range = args.graph_init_range
        self.do_init_rn = args.graph_init_rn
        self.do_init_identity = args.graph_init_identity

        k = args.graph_k
        n_type = args.graph_n_type
        n_head = args.graph_num_relation
        n_basis = args.graph_num_basis
        n_layer = args.graph_gnn_layer_num
        sent_dim = args.text_out_dim
        diag_decompose = args.graph_diag_decompose
        n_attention_head = args.graph_att_head_num
        att_dim = args.graph_att_dim
        att_layer_num = args.graph_att_layer_num
        p_emb = args.graph_dropouti
        p_gnn = args.graph_dropoutg
        p_fc = args.graph_dropoutf
        eps = args.graph_eps
        fc_dim = args.graph_fc_dim
        n_fc_layer = args.graph_fc_layer_num

        self.concept_emb = concept_emb
        concept_dim = self.concept_emb.emb.embedding_dim

        self.gnn = GraphRelationEncoder(
            k=k,
            n_type=n_type,
            n_head=n_head,
            n_basis=n_basis,
            n_layer=n_layer,
            input_size=concept_dim,
            hidden_size=concept_dim,
            sent_dim=sent_dim,
            att_dim=att_dim,
            att_layer_num=att_layer_num,
            dropout=p_gnn,
            diag_decompose=diag_decompose,
            eps=eps,
            ablation=[]
        )

        self.pooler = MultiheadAttPoolLayer(n_attention_head, sent_dim, concept_dim)
        self.fc = MLP(concept_dim + sent_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)
        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)

        if self.init_range > 0:
            self.apply(self._init_weights)

    def _init_rn(self, module):
        if hasattr(module, 'typed_transform'):
            h_size = module.typed_transform.out_features
            half_h_size = h_size // 2
            bias = module.typed_transform.bias
            new_bias = bias.data.clone().detach().view(-1, h_size)
            new_bias[:, :half_h_size] = 1
            bias.data.copy_(new_bias.view(-1))

    def _init_identity(self, module):
        if module.diag_decompose:
            module.w_vs.data[:, :, -1] = 1
        elif module.n_basis == 0:
            module.w_vs.data[:, -1, :, :] = torch.eye(module.w_vs.size(-1), device=module.w_vs.device)
        else:
            print('Warning: init_identity not implemented for n_basis > 0')
            pass

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, MultiHopMessagePassingLayer):
            if 'fix_scale' in self.ablation:
                module.w_vs.data.normal_(mean=0.0, std=np.sqrt(np.pi / 2))
            else:
                module.w_vs.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'w_vs_co'):
                getattr(module, 'w_vs_co').data.fill_(1.0)
            if self.do_init_identity:
                self._init_identity(module)
        elif isinstance(module, PathAttentionLayer):
            if hasattr(module, 'trans_scores'):
                getattr(module, 'trans_scores').data.zero_()
        elif isinstance(module, GraphRelationLayer) and self.do_init_rn:
            self._init_rn(module)

    def decode(self):
        bs, _, n_node, _ = self.adj.size()
        end_ids = self.pool_attn.view(-1, bs, n_node)[0, :, :].argmax(
            -1)  # use only the first head if multi-head attention
        path_ids, path_lengths = self.gnn.decode(end_ids, self.adj)

        # translate local entity ids (0~200) into global eneity ids (0~7e5)
        entity_ids = path_ids[:, ::2]  # (bs, ?)
        path_ids[:, ::2] = self.concept_ids.gather(1, entity_ids)
        return path_ids, path_lengths

    def forward(self, batch, sent_vecs, emb_data=None, concept_embs=None, cache_output=False, bypass_logits=False,
                end2end=False, return_raw_attn=False, goccl=False):
        """
        sent_vecs: (batch_size, d_sent)
        concept_ids: (batch_size, n_node)
        adj: (batch_size, n_head, n_node, n_node)
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)
            0 == question node; 1 == answer node: 2 == intermediate node
        returns: (batch_size, 1)
        """

        concept_ids = batch['concept_ids']
        adj = batch['adj']
        adj_lengths = batch['adj_len']
        node_type_ids = batch['node_type_ids']

        assert emb_data is None, 'Fix the `use_concept_embs` path for ig accoridingly!'

        gnn_input = self.dropout_e(self.concept_emb(concept_ids, emb_data))

        if self.args.save_saliency and self.args.saliency_mode == 'fine' and self.args.saliency_method == 'grad':
            gnn_input.requires_grad = True
        gnn_output = self.gnn(sent_vecs, gnn_input, adj, node_type_ids, cache_output=cache_output)

        mask = torch.arange(concept_ids.size(1), device=adj.device) >= adj_lengths.unsqueeze(1)
        if 'pool_qc' in self.ablation:
            mask = mask | (node_type_ids != 0)
        elif 'pool_all' in self.ablation:
            mask = mask
        else:  # default is to perform pooling over all the answer concepts (pool_ac)
            mask = mask | (node_type_ids != 1)

        if self.args.saliency_mode == 'fine' and self.args.saliency_source == 'target' and self.args.task == 'qa' and self.args.save_saliency == False:
            assert mask.shape == batch['saliency_results'].shape
            mask = mask | ~batch['saliency_results']
        mask[mask.all(1), 0] = 0  # a temporary solution to avoid zero node

        sent_vecs_for_pooler = sent_vecs
        graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, mask, return_raw_attn)

        if bypass_logits:
            node_mask = torch.arange(concept_ids.size(1), device=adj.device) < adj_lengths.unsqueeze(1)
            return gnn_output, node_mask, graph_vecs, pool_attn

        if cache_output:  # cache for decoding
            self.concept_ids = concept_ids
            self.adj = adj
            self.pool_attn = pool_attn

        concat = self.dropout_fc(torch.cat((graph_vecs, sent_vecs), 1))
        logits = self.fc(concat)

        if goccl:
            baseline_vecs = torch.zeros_like(graph_vecs).to(graph_vecs.device)
            baseline_concat = self.dropout_fc(torch.cat((baseline_vecs, sent_vecs), 1))
            baseline_logits = self.fc(baseline_concat)
            return logits, baseline_logits

        if end2end:
            node_mask = torch.arange(concept_ids.size(1), device=adj.device) < adj_lengths.unsqueeze(1)
            # return logits for sal model's f1 calc
            return gnn_output, node_mask, graph_vecs, pool_attn, logits

        if self.args.save_saliency and self.args.saliency_method == 'grad':
            if self.args.saliency_mode == 'coarse':
                sal_units = graph_vecs
            elif self.args.saliency_mode == 'fine':
                sal_units = gnn_input
            return logits, pool_attn, sal_units

        return logits, pool_attn, None


class GraphRelationEncoder(nn.Module):
    def __init__(self, k, n_type, n_head, n_basis, n_layer, input_size, hidden_size, sent_dim,
                 att_dim, att_layer_num, dropout, diag_decompose, eps=1e-20, ablation=None):
        super().__init__()
        self.layers = nn.ModuleList([GraphRelationLayer(k=k, n_type=n_type, n_head=n_head, n_basis=n_basis,
                                                        input_size=input_size, hidden_size=hidden_size,
                                                        output_size=input_size,
                                                        sent_dim=sent_dim, att_dim=att_dim, att_layer_num=att_layer_num,
                                                        dropout=dropout, diag_decompose=diag_decompose, eps=eps,
                                                        ablation=ablation) for _ in range(n_layer)])

    def decode(self, end_ids, A):
        bs = end_ids.size(0)
        k = self.layers[0].message_passing.k
        full_path_ids = end_ids.new_zeros((bs, k * 2 * len(self.layers) + 1))
        full_path_ids[:, 0] = end_ids
        full_path_lengths = end_ids.new_ones((bs,))
        for layer in self.layers[::-1]:
            path_ids, path_lengths = layer.decode(end_ids, A)
            for i in range(bs):
                prev_l = full_path_lengths[i]
                inc_l = path_lengths[i]
                path = path_ids[i]
                assert full_path_ids[i, prev_l - 1] == path[inc_l - 1]
                full_path_ids[i, prev_l:prev_l + inc_l - 1] = path_ids[i, :inc_l - 1].flip((0,))
                full_path_lengths[i] = prev_l + inc_l - 1
        for i in range(bs):
            full_path_ids[i, :full_path_lengths[i]] = full_path_ids[i, :full_path_lengths[i]].flip((0,))
        return full_path_ids, full_path_lengths

    def forward(self, S, H, A, node_type_ids, cache_output=False):
        """
        S: tensor of shape (batch_size, d_sent)
            sentence vectors from an encoder
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: tensor of shape (batch_size, n_head, n_node, n_node)
            adjacency matrices, if A[:, :, i, j] == 1 then message passing from j to i is allowed
        node_type_ids: long tensor of shape (batch_size, n_node)
            0 == question node; 1 == answer node: 2 == intermediate node
        """
        for layer in self.layers:
            H = layer(S, H, A, node_type_ids, cache_output=cache_output)
        return H


class GraphRelationLayer(nn.Module):
    def __init__(self, k, n_type, n_head, n_basis, input_size, hidden_size, output_size, sent_dim,
                 att_dim, att_layer_num, dropout=0.1, diag_decompose=False, eps=1e-20, ablation=None):
        super().__init__()
        assert input_size == output_size
        self.ablation = ablation

        if 'no_typed_transform' not in self.ablation:
            self.typed_transform = TypedLinear(input_size, hidden_size, n_type)
        else:
            assert input_size == hidden_size

        self.path_attention = PathAttentionLayer(n_type, n_head, sent_dim, att_dim, att_layer_num, dropout,
                                                 ablation=ablation)
        self.message_passing = MultiHopMessagePassingLayer(k, n_head, hidden_size, diag_decompose, n_basis, eps=eps,
                                                           ablation=ablation)
        self.aggregator = Aggregator(sent_dim, hidden_size, ablation=ablation)

        self.Vh = nn.Linear(input_size, output_size)
        self.Vz = nn.Linear(hidden_size, output_size)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def decode(self, end_ids, A):
        ks = self.len_attn.argmax(2)  # (bs, n_node)
        if 'detach_s_agg' not in self.ablation:
            ks = ks + 1
        ks = ks.gather(1, end_ids.unsqueeze(-1)).squeeze(-1)  # (bs,)
        path_ids, path_lenghts = self.message_passing.decode(end_ids, ks, A, self.start_attn, self.uni_attn,
                                                             self.trans_attn)
        return path_ids, path_lenghts

    def forward(self, S, H, A, node_type, cache_output=False):
        """
        S: tensor of shape (batch_size, d_sent)
            sentence vectors from an encoder
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: tensor of shape (batch_size, n_head, n_node, n_node)
            adjacency matrices, if A[:, :, i, j] == 1 then message passing from j to i is allowed
        node_type: long tensor of shape (batch_size, n_node)
            0 == question node; 1 == answer node: 2 == intermediate node
        """

        if 'no_typed_transform' not in self.ablation:
            X = self.typed_transform(H, node_type)
        else:
            X = H

        start_attn, end_attn, uni_attn, trans_attn = self.path_attention(S, node_type)

        Z_all = self.message_passing(X, A, start_attn, end_attn, uni_attn, trans_attn)
        Z_all = torch.stack(Z_all, 2)  # (bs, n_node, k, h_size) or (bs, n_node, k+1, h_size)
        Z, len_attn = self.aggregator(S, Z_all)

        if cache_output:  # cache intermediate ouputs for decoding
            self.start_attn, self.uni_attn, self.trans_attn = start_attn, uni_attn, trans_attn
            self.len_attn = len_attn  # (bs, n_node, k)

        if 'early_relu' in self.ablation:
            output = self.Vh(H) + self.activation(self.Vz(Z))
        else:
            output = self.activation(self.Vh(H) + self.Vz(Z))

        output = self.dropout(output)
        return output


class Aggregator(nn.Module):

    def __init__(self, sent_dim, hidden_size, ablation=[]):
        super().__init__()
        self.ablation = ablation
        self.w_qs = nn.Linear(sent_dim, hidden_size)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (sent_dim + hidden_size)))
        self.temperature = np.power(hidden_size, 0.5)
        self.softmax = nn.Softmax(2)

    def forward(self, S, Z_all):
        """
        S: tensor of shape (batch_size, d_sent)
        Z_all: tensor of shape (batch_size, n_node, k, d_node)
        returns: tensor of shape (batch_size, n_node, d_node)
        """
        if 'detach_s_agg' in self.ablation or 'detach_s_all' in self.ablation:
            S = S.detach()
        S = self.w_qs(S)  # (bs, d_node)
        attn = (S[:, None, None, :] * Z_all).sum(-1)  # (bs, n_node, k)
        if 'no_1hop' in self.ablation:
            if 'agg_self_loop' in self.ablation:
                attn[:, :, 1] = -np.inf
            else:
                attn[:, :, 0] = -np.inf

        attn = self.softmax(attn / self.temperature)
        Z = (attn.unsqueeze(-1) * Z_all).sum(2)
        return Z, attn


class PathAttentionLayer(nn.Module):
    def __init__(self, n_type, n_head, sent_dim, att_dim, att_layer_num, dropout=0.1, ablation=[]):
        super().__init__()
        self.n_head = n_head
        self.ablation = ablation
        if 'no_att' not in self.ablation:
            if 'no_type_att' not in self.ablation:
                self.start_attention = MLP(sent_dim, att_dim, n_type, att_layer_num, dropout, layer_norm=True)
                self.end_attention = MLP(sent_dim, att_dim, n_type, att_layer_num, dropout, layer_norm=True)

            if 'no_unary' not in self.ablation and 'no_rel_att' not in self.ablation:
                self.path_uni_attention = MLP(sent_dim, att_dim, n_head, att_layer_num, dropout, layer_norm=True)

            if 'no_trans' not in self.ablation and 'no_rel_att' not in self.ablation:
                if 'ctx_trans' in self.ablation:
                    self.path_pair_attention = MLP(sent_dim, att_dim, n_head ** 2, 1, dropout, layer_norm=True)
                self.trans_scores = nn.Parameter(torch.zeros(n_head ** 2))

    def forward(self, S, node_type):
        """
        S: tensor of shape (batch_size, d_sent)
        node_type: tensor of shape (batch_size, n_node)
        returns: tensors of shapes (batch_size, n_node) (batch_size, n_node) (batch_size, n_head) (batch_size, n_head, n_head)
        """
        n_head = self.n_head
        bs, n_node = node_type.size()

        if 'detach_s_all' in self.ablation:
            S = S.detach()

        if 'no_att' not in self.ablation and 'no_type_att' not in self.ablation:
            bi = torch.arange(bs).unsqueeze(-1).expand(bs, n_node).contiguous().view(-1)  # [0 ... 0 1 ... 1 ...]
            start_attn = self.start_attention(S)
            if 'q2a_only' in self.ablation:
                start_attn[:, 1] = -np.inf
                start_attn[:, 2] = -np.inf
            start_attn = torch.exp(
                start_attn - start_attn.max(1, keepdim=True)[0])  # softmax trick to avoid numeric overflow
            start_attn = start_attn[bi, node_type.view(-1)].view(bs, n_node)
            end_attn = self.end_attention(S)
            if 'q2a_only' in self.ablation:
                end_attn[:, 0] = -np.inf
                end_attn[:, 2] = -np.inf
            end_attn = torch.exp(end_attn - end_attn.max(1, keepdim=True)[0])
            end_attn = end_attn[bi, node_type.view(-1)].view(bs, n_node)
        else:
            start_attn = torch.ones((bs, n_node), device=S.device)
            end_attn = torch.ones((bs, n_node), device=S.device)

        if 'no_att' not in self.ablation and 'no_unary' not in self.ablation and 'no_rel_att' not in self.ablation:
            uni_attn = self.path_uni_attention(S).view(bs, n_head)  # (bs, n_head)
            uni_attn = torch.exp(uni_attn - uni_attn.max(1, keepdim=True)[0]).view(bs, n_head)
        else:
            uni_attn = torch.ones((bs, n_head), device=S.device)

        if 'no_att' not in self.ablation and 'no_trans' not in self.ablation and 'no_rel_att' not in self.ablation:
            if 'ctx_trans' in self.ablation:
                trans_attn = self.path_pair_attention(S) + self.trans_scores
            else:
                trans_attn = self.trans_scores.unsqueeze(0).expand(bs, n_head ** 2)
            trans_attn = torch.exp(trans_attn - trans_attn.max(1, keepdim=True)[0])
            trans_attn = trans_attn.view(bs, n_head, n_head)
        else:
            trans_attn = torch.ones((bs, n_head, n_head), device=S.device)
        return start_attn, end_attn, uni_attn, trans_attn


class MultiHopMessagePassingLayer(nn.Module):
    def __init__(self, k, n_head, hidden_size, diag_decompose, n_basis, eps=1e-20, init_range=0.01, ablation=[]):
        super().__init__()
        self.diag_decompose = diag_decompose
        self.k = k
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.n_basis = n_basis
        self.eps = eps
        self.ablation = ablation

        if diag_decompose and n_basis > 0:
            raise ValueError('diag_decompose and n_basis > 0 cannot be True at the same time')

        if diag_decompose:
            self.w_vs = nn.Parameter(
                torch.zeros(k, hidden_size, n_head + 1))  # the additional head is used for the self-loop
            self.w_vs.data.uniform_(-init_range, init_range)
        elif n_basis == 0:
            self.w_vs = nn.Parameter(torch.zeros(k, n_head + 1, hidden_size, hidden_size))
            self.w_vs.data.uniform_(-init_range, init_range)
        else:
            self.w_vs = nn.Parameter(torch.zeros(k, n_basis, hidden_size * hidden_size))
            self.w_vs.data.uniform_(-init_range, init_range)
            self.w_vs_co = nn.Parameter(torch.zeros(k, n_head + 1, n_basis))
            self.w_vs_co.data.uniform_(-init_range, init_range)

    def _get_weights(self):
        if self.diag_decompose:
            W, Wi = self.w_vs[:, :, :-1], self.w_vs[:, :, -1]
        elif self.n_basis == 0:
            W, Wi = self.w_vs[:, :-1, :, :], self.w_vs[:, -1, :, :]
        else:
            W = self.w_vs_co.bmm(self.w_vs).view(self.k, self.n_head, self.hidden_size, self.hidden_size)
            W, Wi = W[:, :-1, :, :], W[:, -1, :, :]

        k, h_size = self.k, self.hidden_size
        W_pad = [W.new_ones((h_size,)) if self.diag_decompose else torch.eye(h_size, device=W.device)]
        for t in range(k - 1):
            if self.diag_decompose:
                W_pad = [Wi[k - 1 - t] * W_pad[0]] + W_pad
            else:
                W_pad = [Wi[k - 1 - t].mm(W_pad[0])] + W_pad
        assert len(W_pad) == k
        return W, W_pad

    def decode(self, end_ids, ks, A, start_attn, uni_attn, trans_attn):
        """
        end_ids: tensor of shape (batch_size,)
        ks: tensor of shape (batch_size,)
        A: tensor of shape (batch_size, n_head, n_node, n_node)
        start_attn: tensor of shape (batch_size, n_node)
        end_attn: tensor of shape (batch_size, n_node)
        uni_attn: tensor of shape (batch_size, n_head)
        trans_attn: tensor of shape (batch_size, n_head, n_head)
        returns: list[tensor of shape (path_len,)]
        """
        bs, n_head, n_node, n_node = A.size()
        assert ((A == 0) | (A == 1)).all()

        path_ids = end_ids.new_zeros((bs, self.k * 2 + 1))
        path_lengths = end_ids.new_zeros((bs,))

        for idx in range(bs):
            back_trace = []
            end_id, k, adj = end_ids[idx], ks[idx], A[idx]
            uni_a, trans_a, start_a = uni_attn[idx], trans_attn[idx], start_attn[idx]

            if (adj[:, end_id, :] == 0).all():  # end_id is not connected to any other node
                path_ids[idx, 0] = end_id
                path_lengths[idx] = 1
                continue

            dp = F.one_hot(end_id, num_classes=n_node).float()  # (n_node,)
            assert 1 <= k <= self.k
            for t in range(k):
                if t == 0:
                    dp = dp.unsqueeze(0).expand(n_head, n_node)
                else:
                    dp = dp.unsqueeze(0) * trans_a.unsqueeze(-1)  # (n_head, n_head, n_node)
                    dp, ptr = dp.max(1)
                    back_trace.append(ptr)  # (n_head, n_node)
                dp = dp.unsqueeze(-1) * adj  # (n_head, n_node, n_node)
                dp, ptr = dp.max(1)
                back_trace.append(ptr)  # (n_head, n_node)
                dp = dp * uni_a.unsqueeze(-1)  # (n_head, n_node)
            dp, ptr = dp.max(0)
            back_trace.append(ptr)  # (n_node,)
            dp = dp * start_a
            dp, ptr = dp.max(0)
            back_trace.append(ptr)  # （)
            assert dp.dim() == 0
            assert len(back_trace) == k + (k - 1) + 2

            # re-construct path from back_trace
            path = end_id.new_zeros((2 * k + 1,))  # (k + 1) entities and k relations
            path[0] = back_trace.pop(-1)
            path[1] = back_trace.pop(-1)[path[0]]
            for p in range(2, 2 * k + 1):
                if p % 2 == 0:  # need to fill a entity id
                    path[p] = back_trace.pop(-1)[path[p - 1], path[p - 2]]
                else:  # need to fill a relation id
                    path[p] = back_trace.pop(-1)[path[p - 2], path[p - 1]]
            assert len(back_trace) == 0
            assert path[-1] == end_id
            path_ids[idx, :2 * k + 1] = path
            path_lengths[idx] = 2 * k + 1

        return path_ids, path_lengths

    def forward(self, X, A, start_attn, end_attn, uni_attn, trans_attn):
        """
        X: tensor of shape (batch_size, n_node, h_size)
        A: tensor of shape (batch_size, n_head, n_node, n_node)
        start_attn: tensor of shape (batch_size, n_node)
        end_attn: tensor of shape (batch_size, n_node)
        uni_attn: tensor of shape (batch_size, n_head)
        trans_attn: tensor of shape (batch_size, n_head, n_head)
        """
        k, n_head = self.k, self.n_head
        bs, n_node, h_size = X.size()

        W, W_pad = self._get_weights()  # (k, h_size, n_head) or (k, n_head, h_size h_size)

        A = A.view(bs * n_head, n_node, n_node)
        uni_attn = uni_attn.view(bs * n_head)

        Z_all = []
        Z = X * start_attn.unsqueeze(2)  # (bs, n_node, h_size)
        for t in range(k):
            if t == 0:  # Z.size() == (bs, n_node, h_size)
                Z = Z.unsqueeze(-1).expand(bs, n_node, h_size, n_head)
            else:  # Z.size() == (bs, n_head, n_node, h_size)
                Z = Z.permute(0, 2, 3, 1).view(bs, n_node * h_size, n_head)
                Z = Z.bmm(trans_attn).view(bs, n_node, h_size, n_head)
            if self.diag_decompose:
                Z = Z * W[t]  # (bs, n_node, h_size, n_head)
                Z = Z.permute(0, 3, 1, 2).contiguous().view(bs * n_head, n_node, h_size)
            else:
                Z = Z.permute(3, 0, 1, 2).view(n_head, bs * n_node, h_size)
                Z = Z.bmm(W[t]).view(n_head, bs, n_node, h_size)
                Z = Z.permute(1, 0, 2, 3).contiguous().view(bs * n_head, n_node, h_size)
            Z = Z * uni_attn[:, None, None]
            Z = A.bmm(Z)
            Z = Z.view(bs, n_head, n_node, h_size)
            Zt = Z.sum(1) * W_pad[t] if self.diag_decompose else Z.sum(1).matmul(W_pad[t])
            Zt = Zt * end_attn.unsqueeze(2)
            Z_all.append(Zt)

        # compute the normalization factor
        D_all = []
        D = start_attn
        for t in range(k):
            if t == 0:  # D.size() == (bs, n_node)
                D = D.unsqueeze(1).expand(bs, n_head, n_node)
            else:  # D.size() == (bs, n_head, n_node)
                D = D.permute(0, 2, 1).bmm(trans_attn)  # (bs, n_node, n_head)
                D = D.permute(0, 2, 1)
            D = D.contiguous().view(bs * n_head, n_node, 1)
            D = D * uni_attn[:, None, None]
            D = A.bmm(D)
            D = D.view(bs, n_head, n_node)
            Dt = D.sum(1) * end_attn
            D_all.append(Dt)

        Z_all = [Z / (D.unsqueeze(2) + self.eps) for Z, D in zip(Z_all, D_all)]
        assert len(Z_all) == k
        if 'agg_self_loop' in self.ablation:
            Z_all = [X] + Z_all
        return Z_all
