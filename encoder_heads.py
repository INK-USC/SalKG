import torch
from torch import nn
from torch.nn.init import xavier_normal_

from layers import MLP_factory


class BOSTokenLinear(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear = nn.Linear(args.text_in_dim, args.text_out_dim)
        self.encoder_pooler = getattr(args, 'encoder_pooler', 'cls')
        # in case saved ckpt don't have this arg
        if not getattr(args, "no_xavier", False):
            self.initialize(args)

    def forward(self, x, attn_mask, **unused):
        if self.encoder_pooler == 'cls':
            sent_vecs = x[:, 0, :]
        elif self.encoder_pooler == 'mean':
            sent_vecs = (x * attn_mask.unsqueeze(-1)).sum(1) / attn_mask.sum(1).unsqueeze(-1)

        return self.linear(sent_vecs)

    def initialize(self, args):
        torch.manual_seed(args.seed)
        xavier_normal_(self.linear.weight)


class BOSTokenMLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder_pooler = getattr(args, 'encoder_pooler', 'cls')
        layer_sizes = [[args.text_in_dim, 1], [args.text_hidden_dim, args.text_hidden_layers], [args.text_out_dim, 1]]
        self.mlp = MLP_factory(layer_sizes, dropout=args.text_dropout, layer_norm=args.text_layer_norm)
        # in case saved ckpt don't have this arg
        if not getattr(args, "no_xavier", False):
            self.initialize(args)

    def forward(self, x, attn_mask, **unused):
        if self.encoder_pooler == 'cls':
            sent_vecs = x[:, 0, :]
        elif self.encoder_pooler == 'mean':
            sent_vecs = (x * attn_mask.unsqueeze(-1)).sum(1) / attn_mask.sum(1).unsqueeze(-1)

        return self.mlp(sent_vecs)

    def initialize(self, args):
        torch.manual_seed(args.seed)
        for layer in self.mlp:
            if type(layer) == torch.nn.Linear:
                xavier_normal_(layer.weight)


class BOSTokenDirect(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder_pooler = getattr(args, 'encoder_pooler', 'cls')

    def forward(self, x, attn_mask, **unused):
        if self.encoder_pooler == 'cls':
            sent_vecs = x[:, 0, :]
        elif self.encoder_pooler == 'mean':
            sent_vecs = (x * attn_mask.unsqueeze(-1)).sum(1) / attn_mask.sum(1).unsqueeze(-1)

        return sent_vecs
