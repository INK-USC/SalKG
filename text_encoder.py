import torch
from transformers import AlbertModel, RobertaModel, BertModel

from encoder_heads import BOSTokenDirect, BOSTokenLinear, BOSTokenMLP

arch_dict = {
    'albert_xxlarge_v2': AlbertModel.from_pretrained('albert-xxlarge-v2'),
    'roberta_base': RobertaModel.from_pretrained('roberta-base'),
    'roberta_large': RobertaModel.from_pretrained('roberta-large'),
    'bert_base_uncased': BertModel.from_pretrained('bert-base-uncased'),
    'bert_large_uncased': BertModel.from_pretrained('bert-large-uncased'),
    'aristoroberta_large': RobertaModel.from_pretrained('roberta-large'),
}

encoder_head_dict = {
    'bos_token_linear': BOSTokenLinear,
    'bos_token_mlp': BOSTokenMLP,
    'bos_token_direct': BOSTokenDirect,
}


class TextEncoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder_head = args.text_encoder_head
        self.model = arch_dict[args.arch]
        if args.arch == 'aristoroberta_large':
            self.load_aristo_checkpoint(args.aristo_path)
        self.pad_idx = self.model.config.pad_token_id
        if self.encoder_head != 'pooler_output':
            self.output_layer = encoder_head_dict[self.encoder_head](args)

    def load_aristo_checkpoint(self, path):
        print('Loading weights for AristoRoberta...')
        assert path is not None
        weight = torch.load(path, map_location='cpu')
        new_dict = {}
        for k, v in weight.items():
            nk = k.replace('_transformer_model.', '')
            if nk not in self.state_dict():
                print(k)
                continue
            new_dict[nk] = v
        model_dict = self.state_dict()
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, masked_tokens=None, **unused):
        x = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)
        if not (features_only or self.encoder_head == 'pooler_output'):
            x = self.output_layer(x, attn_mask=src_tokens.ne(self.pad_idx), masked_tokens=masked_tokens)
        return x

    def extract_features(self, src_tokens, **kwargs):
        output = self.model(input_ids=src_tokens, attention_mask=src_tokens.ne(self.pad_idx))
        if self.encoder_head == 'pooler_output':
            features = output['pooler_output']
        else:
            features = output['last_hidden_state']
        return features

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.model.config.max_position_embeddings
