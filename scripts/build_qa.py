import json
import os
import sys

sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np
import torch, time
import argparse
import pickle5 as pickle
import tqdm

from transformers import (OpenAIGPTTokenizer, BertTokenizer, XLNetTokenizer, RobertaTokenizer, AlbertTokenizer)
from transformers import (OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP)
import indexed_dataset
from constants import TRAINING_TQDM_BAD_FORMAT, ADJ_KEYS

MODEL_CLASS_TO_NAME = {
    'gpt': list(OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'bert': list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'xlnet': list(XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'roberta': list(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'albert': list(ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'lstm': ['lstm'],
}
MODEL_NAME_TO_CLASS = {model_name: model_class for model_class, model_name_list in MODEL_CLASS_TO_NAME.items() for
                       model_name in model_name_list}

TOKENIZER_DICT = {
    'gpt': OpenAIGPTTokenizer,
    'bert': BertTokenizer,
    'xlnet': XLNetTokenizer,
    'roberta': RobertaTokenizer,
    'albert': AlbertTokenizer
}


def load_input_tensors(statement_jsonl_path, model_type, model_name, max_seq_length, format=[]):
    class InputExample(object):
        def __init__(self, example_id, question, contexts, endings, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.label = label

    class InputFeatures(object):
        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                }
                for _, input_ids, input_mask, segment_ids, output_mask in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            for line in f.readlines():
                json_dic = json.loads(line)
                label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else 0
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=[json_dic["question"]["stem"]] * len(json_dic["question"]["choices"]),
                        question="",
                        endings=[ending["text"] for ending in json_dic["question"]["choices"]],
                        label=label
                    ))
        return examples

    def convert_examples_to_features(examples, label_list, max_seq_length,
                                     tokenizer,
                                     cls_token_at_end=False,
                                     cls_token='[CLS]',
                                     cls_token_segment_id=1,
                                     sep_token='[SEP]',
                                     sequence_a_segment_id=0,
                                     sequence_b_segment_id=1,
                                     sep_token_extra=False,
                                     pad_token_segment_id=0,
                                     pad_on_left=False,
                                     pad_token=0,
                                     mask_padding_with_zero=True):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for ex_index, example in enumerate(examples):
            choices_features = []
            for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
                extra_args = {'add_prefix_space': True} if (
                            model_type in ['roberta'] and 'add_prefix_space' in format) else {}
                tokens_a = tokenizer.tokenize(context, **extra_args)
                tokens_b = tokenizer.tokenize(example.question + " " + ending, **extra_args)

                special_tokens_count = 4 if sep_token_extra else 3
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)

                # The convention in BERT is:
                # (a) For sequence pairs:
                #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
                #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
                # (b) For single sequences:
                #  tokens:   [CLS] the dog is hairy . [SEP]
                #  type_ids:   0   0   0   0  0     0   0
                #
                # Where "type_ids" are used to indicate whether this is the first
                # sequence or the second sequence. The embedding vectors for `type=0` and
                # `type=1` were learned during pre-training and are added to the wordpiece
                # embedding vector (and position vector). This is not *strictly* necessary
                # since the [SEP] token unambiguously separates the sequences, but it makes
                # it easier for the model to learn the concept of sequences.
                #
                # For classification tasks, the first vector (corresponding to [CLS]) is
                # used as as the "sentence vector". Note that this only makes sense because
                # the entire model is fine-tuned.
                tokens = tokens_a + [sep_token]
                if sep_token_extra:
                    # roberta uses an extra separator b/w pairs of sentences
                    tokens += [sep_token]

                segment_ids = [sequence_a_segment_id] * len(tokens)

                if tokens_b:
                    tokens += tokens_b + [sep_token]
                    segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

                if cls_token_at_end:
                    tokens = tokens + [cls_token]
                    segment_ids = segment_ids + [cls_token_segment_id]
                else:
                    tokens = [cls_token] + tokens
                    segment_ids = [cls_token_segment_id] + segment_ids

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.

                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
                special_token_id = tokenizer.convert_tokens_to_ids([cls_token, sep_token])
                output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                    output_mask = ([1] * padding_length) + output_mask

                    segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                else:
                    input_ids = input_ids + ([pad_token] * padding_length)
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    output_mask = output_mask + ([1] * padding_length)
                    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                choices_features.append((tokens, input_ids, input_mask, segment_ids, output_mask))
            label = label_map[example.label]
            features.append(
                InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.uint8)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label

    tokenizer = TOKENIZER_DICT[model_type].from_pretrained(model_name)

    examples = read_examples(statement_jsonl_path)
    if any(x in format for x in ('add_qa_prefix', 'fairseq')):
        for example in examples:
            example.contexts = ['Q: ' + c for c in example.contexts]
            example.endings = ['A: ' + e for e in example.endings]

    features = convert_examples_to_features(examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer,
                                            cls_token_at_end=bool(model_type in ['xlnet']),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(model_type in ['roberta']
                                                                 and 'no_extra_sep' not in format
                                                                 and 'fairseq' not in format),
                                            cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
                                            pad_on_left=bool(model_type in ['xlnet']),  # pad on the left for xlnet
                                            pad_token=tokenizer.pad_token_id or 0,
                                            pad_token_segment_id=4 if model_type in ['xlnet'] else 0,
                                            sequence_b_segment_id=0 if model_type in ['roberta'] else 1)

    example_ids = [f.example_id for f in features]
    all_input_ids, _, _, _, all_label = convert_features_to_tensors(features)

    return (example_ids, all_input_ids, all_label)


def load_2hop_relational_paths(rpath_jsonl_path,
                               max_tuple_num=200, num_choice=None):
    """
    rpath: relpath.2hop
      each row a relpath dict. 
        key[acs -> list[str] answer text, 
            qcs -> list] question text hit in conceptnet,
            paths -> list of dict{ac qc id pair & list of rel(qc, ac) at most 2 hop}
    
    total rows=n_samples = all samples * num_choices
    ------------------------------
    return
    qa_data (N, num_choices, max_tuple_num, 2)
    rel_data (N, num_choices, max_tuple_num)
    num_tuples: (N, num_choices, )

    for ith sample, if contains k (qc, ac) pairs in rpath[i]['path']
      qa_data first k rows contains the two concept ids
      rel_data first k rows = each of k's 2hop relation id 
        if 1 relation, = rel id; if 2 = 34 + rel0 * 34 + rel1
      num_tuples ith = k
    """
    with open(rpath_jsonl_path, 'r') as fin:
        rpath_data = [json.loads(line) for line in fin]

        # N * num_choices
    n_samples = len(rpath_data)
    qa_data = torch.zeros((n_samples, max_tuple_num, 2), dtype=torch.long)
    rel_data = torch.zeros((n_samples, max_tuple_num), dtype=torch.long)
    num_tuples = torch.zeros((n_samples,), dtype=torch.long)

    for i, data in enumerate(tqdm.tqdm(rpath_data, total=n_samples, desc='loading QA pairs')):
        cur_qa = []
        cur_rel = []
        for dic in data['paths']:
            if len(dic['rel']) == 1:
                cur_qa.append([dic['qc'], dic['ac']])
                cur_rel.append(dic['rel'][0])
            elif len(dic['rel']) == 2:
                cur_qa.append([dic['qc'], dic['ac']])
                cur_rel.append(34 + dic['rel'][0] * 34 + dic['rel'][1])
            else:
                raise ValueError('Invalid path length, must be 2 hops')
            if len(cur_qa) >= max_tuple_num:
                # cap at max_tuple_num
                break
        assert len(cur_qa) == len(cur_rel)

        if len(cur_qa) > 0:
            qa_data[i][:len(cur_qa)] = torch.tensor(cur_qa)
            rel_data[i][:len(cur_rel)] = torch.tensor(cur_rel)
            num_tuples[i] = (len(cur_qa) + len(cur_rel)) // 2  # code style suggested by kiwisher

    if num_choice is not None:
        qa_data = qa_data.view(-1, num_choice, max_tuple_num, 2)
        rel_data = rel_data.view(-1, num_choice, max_tuple_num)
        num_tuples = num_tuples.view(-1, num_choice)

    flat_rel_data = rel_data.view(-1, max_tuple_num)
    flat_num_tuples = num_tuples.view(-1)
    valid_mask = (torch.arange(max_tuple_num) < flat_num_tuples.unsqueeze(-1)).float()
    n_1hop_paths = ((flat_rel_data < 34).float() * valid_mask).sum(1)
    n_2hop_paths = ((flat_rel_data >= 34).float() * valid_mask).sum(1)
    print('| #paths: {} | average #1-hop paths: {} | average #2-hop paths: {} | #w/ 1-hop {} | #w/ 2-hop {} |'.format(
        flat_num_tuples.float().mean(0), n_1hop_paths.mean(), n_2hop_paths.mean(),
        (n_1hop_paths > 0).float().mean(), (n_2hop_paths > 0).float().mean()))
    return qa_data, rel_data, num_tuples


def load_adj_data(adj_pk_path, max_node_num, num_choice):
    """
    adj_pk_path: graph.adj
      total rows = N * num_choices
      each row i 4-tuple
      [adj, (#rel * ci, ci), originally adj matrix (#rel, #node, #node), reshpae to COO (-1, #node)
       concepts ids, (ci, ) node ids, nodes related to ith sample either question or answer
       qm, (ci) bool mask, among these ci nodes, which are question
       am, (ci) bool mask, among these ci nodes, which are answer
    ------------------------------
    return
    concept_ids (N, num_choices, max_node_num, 2)
    node_type_ids (N, num_choices, max_node_num)
       0 == question node; 1 == answer node: 2 == intermediate node
    adj_lengths: (N, num_choices, )
    adj_data: (N, num_choices, 3) i, j, k index of which adj = 1
      all tensors same shape = #adj=1

    if ith sample has ci concepts ids
      concept_ids first ci row = their ids
      node_type_ids first ci row: based on each of ci's node type
      adj_lengths ith = ci
      adj_data append i j k index of which adj = 1
    """
    with open(adj_pk_path, 'rb') as fin:
        adj_concept_pairs = pickle.load(fin)

    n_samples = len(adj_concept_pairs)
    adj_data = []
    adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
    concept_ids = torch.zeros((n_samples, max_node_num), dtype=torch.long)
    node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long)

    adj_lengths_ori = adj_lengths.clone()
    for idx, (adj, concepts, qm, am) in tqdm.tqdm(enumerate(adj_concept_pairs), total=n_samples,
                                                  desc='loading adj matrices'):
        num_concept = min(len(concepts), max_node_num)
        adj_lengths_ori[idx] = len(concepts)
        concepts = concepts[:num_concept]
        concept_ids[idx, :num_concept] = torch.tensor(concepts)  # note : concept zero padding is disabled

        adj_lengths[idx] = num_concept
        node_type_ids[idx, :num_concept][torch.tensor(qm, dtype=torch.uint8)[:num_concept]] = 0
        node_type_ids[idx, :num_concept][torch.tensor(am, dtype=torch.uint8)[:num_concept]] = 1
        ij = torch.tensor(adj.row, dtype=torch.int64)  # row of which adj = 1
        k = torch.tensor(adj.col, dtype=torch.int64)  # col of which adj = 1
        n_node = adj.shape[1]
        half_n_rel = adj.shape[0] // n_node
        i, j = ij // n_node, ij % n_node
        mask = (j < max_node_num) & (k < max_node_num)
        i, j, k = i[mask], j[mask], k[mask]

        i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j), 0)  # add inverse relations
        adj_data.append((i, j, k))  # i, j, k are the coordinates of adj's non-zero entries

    print('| ori_adj_len: {:.2f} | adj_len: {:.2f} |'.format(adj_lengths_ori.float().mean().item(),
                                                             adj_lengths.float().mean().item()) +
          ' prune_rate： {:.2f} |'.format((adj_lengths_ori > adj_lengths).float().mean().item()) +
          ' qc_num: {:.2f} | ac_num: {:.2f} |'.format((node_type_ids == 0).float().sum(1).mean().item(),
                                                      (node_type_ids == 1).float().sum(1).mean().item()))
    # reshape 1st dim n_samples -> N * num_choices
    concept_ids, node_type_ids, adj_lengths = [x.view(-1, num_choice, *x.size()[1:]) for x in
                                               (concept_ids, node_type_ids, adj_lengths)]
    adj_data = list(map(list, zip(*(iter(adj_data),) * num_choice)))

    return concept_ids, node_type_ids, adj_lengths, adj_data


def main(args):
    # random permutation for training set split
    np.random.seed(0)

    assert args.split != None and args.model_name != None
    model_name = 'roberta-large' if args.model_name == 'aristoroberta-large' else args.model_name
    model_type = MODEL_NAME_TO_CLASS[model_name]

    # Get preprocessed data (via MHGRN repo)
    inhouse = args.inhouse if args.dataset in ['csqa', 'qasc'] else False
    # if args.dataset == 'csqa' and not inhouse:
    #     raise NotImplementedError

    if inhouse:
        data_split = 'train' if args.split != 'valid' else 'dev'
    else:
        data_split = args.split if args.split != 'valid' else 'dev'

    if args.model_name == 'aristoroberta-large':
        assert args.dataset == 'obqa'

    # qasc needs 2-step IR
    if args.dataset == "qasc":
        statement_path = os.path.join(args.root_dir, 'mhgrn_data', args.dataset, '{}_2step.jsonl'.format(data_split))
    elif args.dataset == 'obqa' and args.model_name == 'aristoroberta-large':
        args.max_seq_len = 128
        statement_path = os.path.join(args.root_dir, 'mhgrn_data', args.dataset, 'AristoEvidence',
                                      'roberta_{}.jsonl'.format(data_split))
    else:
        statement_path = os.path.join(args.root_dir, 'mhgrn_data', args.dataset, 'statement',
                                      '{}.statement.jsonl'.format(data_split))
    adj_path = os.path.join(args.root_dir, 'mhgrn_data', args.dataset, 'graph', '{}.graph.adj.pk'.format(data_split))
    # (N, ), (N, num_choices, max_seq_len), (N, )
    qids, text_input_ids, labels = load_input_tensors(statement_path, model_type, model_name, args.max_seq_len,
                                                      format=[])
    num_choice = text_input_ids.size(1)

    # Load RN-specific data
    rpath_jsonl = os.path.join(args.root_dir, 'mhgrn_data', args.dataset, 'paths',
                               '{}.relpath.2hop.jsonl'.format(data_split))
    qa_ids, rel_ids, num_tuples = load_2hop_relational_paths(rpath_jsonl, max_tuple_num=args.max_tuple_num,
                                                             num_choice=num_choice)

    # Load MHGRN-specific data
    concept_ids, node_type_ids, adj_lengths, adj_data = load_adj_data(adj_path, args.max_node_num, num_choice)

    # Load PathGen-specific data
    path_embedding_path = os.path.join(args.root_dir, args.dataset, 'path_embedding.pickle')
    with open(path_embedding_path, 'rb') as handle:
        path_embedding = pickle.load(handle)
    # (N, num_choices, num_choices, 768)
    path_emb = path_embedding[data_split]
    if args.fine_occl:
        path_emb_fine_occl = []

    assert all(len(qids) == len(adj_data) == x.size(0) for x in
               [labels] + [text_input_ids] + [qa_ids, rel_ids, num_tuples] + [concept_ids, node_type_ids,
                                                                              adj_lengths] + [path_emb])

    if args.split in ['train', 'test'] and inhouse:
        inhouse_train_qids_path = os.path.join(args.root_dir, f'mhgrn_data/{args.dataset}/inhouse_split_qids.txt')
        with open(inhouse_train_qids_path, 'r') as fin:
            inhouse_qids = set(line.strip() for line in fin)
        if args.split == 'train':
            indices = np.array([i for i, qid in enumerate(qids) if qid in inhouse_qids])
        else:
            indices = np.array([i for i, qid in enumerate(qids) if qid not in inhouse_qids])
    else:
        indices = np.arange(len(qids))

    # Training set split for saliency task
    if args.split == 'train' and args.train_percentage < 100:
        ind_permutation = np.random.permutation(len(indices))
        chosen = ind_permutation[:int(len(ind_permutation) / 100 * args.train_percentage)]
        indices = indices[chosen]

    bin_suffix = '' if args.train_percentage == 100 else str(args.train_percentage)

    # Create output dir
    if inhouse:
        split_type = 'inhouse'
    elif args.dataset == "codah":
        split_type = "fold_0"
    else:
        split_type = "official"

    output_dir = os.path.join(args.root_dir, args.dataset, split_type, args.model_name, 'bin' + bin_suffix)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create indexed dataset builders
    dataset_builders = {}
    rn_keys = ['qa', 'rel', 'num_tuples']
    mhgrn_keys = ['concept', 'node_type', 'adj_len']
    keys = rn_keys + mhgrn_keys
    if args.fine_occl:
        keys += ['rn_text', 'rn_label', 'rn_id', 'mhgrn_text', 'mhgrn_label', 'mhgrn_id', 'adj']
        output_suffix = 'fine_occl.'
    else:
        keys += ['text', 'label']
        keys += ADJ_KEYS[:num_choice]
        output_suffix = ''

    for key in keys:
        dataset_builders[key] = indexed_dataset.make_builder(
            os.path.join(output_dir, '{}.{}.{}bin'.format(args.split, key, output_suffix)),
            impl="mmap"
        )

    # Initialize progress bar
    pbar = tqdm.tqdm(
        total=len(indices),
        desc='Processing data',
        bar_format=TRAINING_TQDM_BAD_FORMAT,
    )

    # Build indexed datasets
    for i in indices:
        if args.fine_occl:
            # fine occl, instead of saving each as a whole
            # save each indivdual over num_choices & num_tuples/path (rn & pathgen) | num_nodes (mhgrn)
            for j in range(num_choice):
                for k in range(-1, num_tuples[i, j]):
                    qa_ids_ = qa_ids[i, j].clone()
                    rel_ids_ = rel_ids[i, j].clone()

                    if k > -1:
                        qa_ids_[:] = 0
                        qa_ids_[:k] = qa_ids[i, j, :k]
                        qa_ids_[k:-1] = qa_ids[i, j, k + 1:]

                        rel_ids_[:] = 0
                        rel_ids_[:k] = rel_ids[i, j, :k]
                        rel_ids_[k:-1] = rel_ids[i, j, k + 1:]

                    dataset_builders['rn_text'].add_item(text_input_ids[i, j])
                    dataset_builders['rn_label'].add_item(labels[i])
                    dataset_builders['rn_id'].add_item(torch.LongTensor([i, j, k]))
                    dataset_builders['qa'].add_item(qa_ids_)
                    dataset_builders['rel'].add_item(rel_ids_)
                    dataset_builders['num_tuples'].add_item(num_tuples[i, j])
                    path_emb_fine_occl.append(path_emb[i, j])

                for k in range(-1, adj_lengths[i, j]):
                    concept_ids_ = concept_ids[i, j].clone()
                    node_type_ids_ = node_type_ids[i, j].clone()

                    if k == -1:
                        dataset_builders['adj'].add_item(torch.stack(adj_data[i][j]))
                    else:
                        concept_ids_[k] = 0
                        node_type_ids_[k] = 0
                        keep_indices = torch.nonzero((adj_data[i][j][1] != k) * (adj_data[i][j][2] != k)).flatten()
                        dataset_builders['adj'].add_item(torch.stack([x[keep_indices] for x in adj_data[i][j]]))

                    dataset_builders['mhgrn_text'].add_item(text_input_ids[i, j])
                    dataset_builders['mhgrn_label'].add_item(labels[i])
                    dataset_builders['mhgrn_id'].add_item(torch.LongTensor([i, j, k]))
                    dataset_builders['concept'].add_item(concept_ids_)
                    dataset_builders['node_type'].add_item(node_type_ids_)
                    dataset_builders['adj_len'].add_item(adj_lengths[i, j])


        else:
            dataset_builders['text'].add_item(text_input_ids[i])
            dataset_builders['label'].add_item(labels[i])

            dataset_builders['qa'].add_item(qa_ids[i])
            dataset_builders['rel'].add_item(rel_ids[i])
            dataset_builders['num_tuples'].add_item(num_tuples[i])

            dataset_builders['concept'].add_item(concept_ids[i])
            dataset_builders['node_type'].add_item(node_type_ids[i])
            dataset_builders['adj_len'].add_item(adj_lengths[i])
            for j, key in enumerate(ADJ_KEYS[:num_choice]):
                dataset_builders[key].add_item(torch.stack(adj_data[i][j]))

        pbar.update()

    pbar.close()

    # Save path embeddings
    path_emb_path = os.path.join(output_dir, 'path_embedding_{}.{}pickle'.format(args.split, output_suffix))
    if args.fine_occl:
        path_emb = torch.stack(path_emb_fine_occl).unsqueeze(1)
    else:
        path_emb = path_emb[indices]
    print('Saving path embeddings...')
    start_time = time.perf_counter()
    with open(path_emb_path, 'wb') as handle:
        pickle.dump(path_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Finished saving path embeddings in {:.2f} seconds!'.format(time.perf_counter() - start_time))

    # Finalize indexed datasets
    for key in keys:
        dataset_builders[key].finalize(os.path.join(output_dir, '{}.{}.{}idx'.format(args.split, key, output_suffix)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing')
    parser.add_argument('--split', type=str, help='Dataset split', choices=['train', 'valid', 'test'])
    parser.add_argument('--model-name', type=str, default='roberta-large',
                        choices=['bert-base-uncased', 'bert-large-uncased', 'roberta-base', 'roberta-large',
                                 'aristoroberta-large', 'albert-xxlarge-v2'])
    parser.add_argument('--root-dir', type=str, default='../data/', help='Root directory')
    parser.add_argument('--dataset', default='csqa', type=str, choices=['csqa', 'obqa', 'codah', 'qasc'])
    parser.add_argument('--inhouse', default=False, action='store_true')
    parser.add_argument('--max-seq-len', type=int, default=64)
    parser.add_argument('--max-node-num', type=int, default=200)
    parser.add_argument('--max-tuple-num', type=int, default=200)
    parser.add_argument('--fine-occl', default=False, action='store_true')
    parser.add_argument('--train-percentage', type=int, default=100)
    args = parser.parse_args()
    main(args)
