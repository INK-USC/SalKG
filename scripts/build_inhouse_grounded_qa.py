import os;
import sys

sys.path.append(os.path.join(sys.path[0], '..'))
import random
from helper import *


def get_top_k_indices(sal, k, use_pct):
    if use_pct:
        indices = np.argsort(sal)[::-1][:(int)(len(sal) * k / 100)]
    else:
        indices = np.argsort(sal)[::-1][:k]
    return indices


def get_lpg_indices(sal):
    sorted_scores = np.sort(sal)
    diff = np.diff(sorted_scores)
    threshold_idx = np.argmax(diff) if diff.size != 0 else 0
    lpg_threshold = sorted_scores[threshold_idx]

    lpg_indices = np.squeeze(np.argwhere(sal > lpg_threshold), axis=1)
    return lpg_indices


def get_m_dict_from_threshold(sal, m):
    sorted_scores = np.sort(sal)
    diff = np.diff(sorted_scores)
    threshold_idx = np.argmax(diff) if diff.size != 0 else 0
    lpg_threshold = sorted_scores[threshold_idx]
    upper = sorted_scores[min(threshold_idx + m, len(sorted_scores) - 1)]
    lower = sorted_scores[max(threshold_idx - m, 0)]
    btm_sal = np.squeeze(np.argwhere((sal > lpg_threshold) & (sal <= upper)))
    top_unsal = np.squeeze(np.argwhere((sal <= lpg_threshold) & (sal >= lower)))
    return btm_sal, top_unsal


def load_graph_data(adj_pk_path, num_choice):
    with open(adj_pk_path, 'rb') as fin:
        adj_concept_pairs = pickle.load(fin)
    n_samples = len(adj_concept_pairs)
    node_type_ids = torch.full((n_samples, 200), 0, dtype=torch.long)
    concept_ids = []
    for idx, (adj, concepts, qm, am) in tqdm.tqdm(enumerate(adj_concept_pairs), total=n_samples,
                                                  desc='loading adj matrices'):
        num_concept = min(len(concepts), 200)
        concepts = set(concepts[:num_concept])
        concept_ids.append(concepts)
        node_type_ids[idx, :num_concept][torch.tensor(qm, dtype=torch.bool)[:num_concept]] = 1
        node_type_ids[idx, :num_concept][torch.tensor(am, dtype=torch.bool)[:num_concept]] = 1

    node_type_ids = node_type_ids.view(-1, num_choice, *node_type_ids.size()[1:])

    return node_type_ids


def main(args):
    random.seed(0)
    assert args.split != None and args.text_encoder != None and args.graph_encoder != None
    inhouse = args.inhouse if args.dataset == 'csqa' else False
    if args.dataset == 'csqa' and not inhouse:
        raise NotImplementedError
    split_type = 'inhouse' if inhouse else 'official'
    choice_keys = CHOICE_KEYS[:NUM_CHOICES[args.dataset]]

    if args.dataset == 'csqa' and split_type == 'inhouse':
        indices = np.arange(NUM_CSQA_INHOUSE[args.split])
    elif args.dataset == 'obqa' and split_type == 'official':
        indices = np.arange(NUM_OBQA_OFFICIAL[args.split])
    elif args.dataset == 'codah':
        indices = np.arange(NUM_CODAH_FOLD_0[args.split])
    else:
        raise NotImplementedError

    # Create output dir
    split_type = 'inhouse' if args.inhouse else 'official'
    output_dir = os.path.join(args.root_dir, args.dataset, split_type, args.text_encoder, 'bin')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    g_split = 'dev' if args.split == 'valid' else 'train'
    # indices_path = os.path.join(args.root_dir, args.dataset, 'inhouse', args.text_encoder, 'bin', '{}.inhouse_indices.pk'.format(args.split))
    # with open(indices_path, 'rb') as fin:
    #     indices = pickle.load(fin)
    # print(len(indices))

    # Load QA concepts
    # grounded_path = os.path.join(args.root_dir, 'mhgrn_data', args.dataset, 'grounded', '{}.grounded.jsonl'.format(g_split))
    # output_path = os.path.join(args.root_dir, 'mhgrn_data', args.dataset, 'grounded', '{}.grounded_inhouse.jsonl'.format(args.split))
    adj_path = os.path.join(args.root_dir, 'mhgrn_data', args.dataset, 'graph', '{}.graph.adj.pk'.format(g_split))
    node_type_ids = load_graph_data(adj_path, len(choice_keys))
    split_node_types = node_type_ids[indices]
    print(split_node_types.shape)
    output_path = os.path.join(args.root_dir, 'mhgrn_data', args.dataset, 'graph',
                               '{}.node_type_ids.pk'.format(args.split))
    with open(output_path, 'wb') as fout:
        pickle.dump(split_node_types, fout)
    # with open(adj_path, 'rb') as fin, open(output_path, 'wb') as fout:
    #     adj_concept_pairs = pickle.load(fin) 
    #     output_paris = []
    #     for i, data in tqdm.tqdm(enumerate(adj_concept_pairs)):
    #         if args.split == 'train':
    #             if i in indices:
    #                 output_paris.append(data)
    #         elif args.split == 'test':
    #             if i in indices:
    #                 output_paris.append(data)
    #         else:
    #             output_paris.append(data)
    #     pickle.dump(output_paris, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process saliency scores')
    parser.add_argument('--split', type=str, default='train', help='Dataset split', choices=['train', 'valid', 'test'])
    parser.add_argument('--text-encoder', type=str, default='roberta-large',
                        choices=['roberta-base', 'roberta-large', 'albert-xxlarge-v2', 'bert-base-uncased'])
    parser.add_argument('--graph-encoder', type=str, default='mhgrn', choices=['rn', 'mhgrn'])
    parser.add_argument('--root-dir', type=str, default='../data/', help='dataset root directory')
    parser.add_argument('--dataset', default='csqa', type=str, choices=['csqa', 'obqa', 'codah'])
    parser.add_argument('--inhouse', default=True, type=bool)
    parser.add_argument('--saliency-method', type=str, default='grad', choices=['grad', 'occl'])
    parser.add_argument('--saliency-source', type=str, default='target', choices=['pred', 'target'])
    parser.add_argument('--top-k', default=10, type=int)
    parser.add_argument('--usepct', default=False, type=bool)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument(
        '--dataset-impl',
        metavar='FORMAT',
        default='mmap',
        choices=indexed_dataset.get_available_dataset_impl(),
        help='output dataset implementation')
    args = parser.parse_args()
    main(args)
