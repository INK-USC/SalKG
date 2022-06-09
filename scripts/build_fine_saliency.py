import os
import sys

sys.path.append(os.path.join(sys.path[0], '..'))
from pathlib import Path
from collections import defaultdict


def create_dataset(args, output_dir, choice_keys):
    dataset_builders = {}
    filenames = {}
    for key in choice_keys:
        filename = os.path.join(output_dir, '{}.{}.sal_fine_{}_{}_{}_{}_{}_FS{}'.format(
            args.split,
            args.graph_encoder,
            args.saliency_method,
            args.saliency_source,
            args.method,
            args.value,
            key,
            args.exp
        ))

        filenames[key] = filename
        dataset_builders[key] = indexed_dataset.make_builder(filename + '.bin', impl=args.dataset_impl)
    return dataset_builders, filenames


def finalize_dataset(datasets, names):
    print('Indexed Datasets are saved to:')
    for k, ds in datasets.items():
        path = Path(names[k] + '.idx')
        print(path)
        if path.exists():
            print(f"path {path} already exists, rm")
            os.unlink(path)
            os.unlink(Path(names[k] + ".bin"))

        ds.finalize(names[k] + '.idx')


def load_sal_scores(input_path, saliency_method, indices, num_choices):
    dataset_dict = ddict(list)
    pbar = tqdm.tqdm(
        total=len(indices),
        desc='Loading saliency scores',
        bar_format=TRAINING_TQDM_BAD_FORMAT,
    )
    if saliency_method == 'occl':
        instance_idx = -1

    with open(input_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row_arr = np.array([float(x) for x in row])
            if saliency_method in ['grad', 'ig']:
                instance_idx = int(row_arr[0])
                choice_idx = int(row_arr[1])
                scores = row_arr[2:]
                dataset_dict[instance_idx].append(scores)
                if choice_idx == num_choices - 1:
                    pbar.update()
            elif saliency_method == 'occl':
                choice_idx = int(row_arr[2])
                unit_idx = int(row_arr[3])
                target = int(row_arr[4])
                logit = row_arr[5]
                if choice_idx == 0 and unit_idx < 0:
                    instance_idx += 1
                    dataset_dict[instance_idx] = [np.array([]) for _ in range(num_choices)]
                    pbar.update()
                elif unit_idx >= 0:
                    score = -1 * logit if choice_idx == target else logit
                    dataset_dict[instance_idx][choice_idx] = np.append(dataset_dict[instance_idx][choice_idx], score)

    return dataset_dict


def get_target(method, sal, node_type, val):
    if len(sal) <= 1:  # Account for RN/PathGen graphs with no paths
        return np.zeros(1)

    if method == 'random_all':
        saliency_targets = np.random.randint(2, size=len(sal))
    else:
        if method == 'ratio':
            pos_indices = np.argsort(sal)[::-1][:int(len(sal) * val / 100)]
            assert len(pos_indices) == int(len(sal) * val / 100) or len(pos_indices) == len(sal)
        elif method == 'topk':
            pos_indices = np.argsort(sal)[::-1][:int(val)]
            assert len(pos_indices) == int(val) or len(pos_indices) == len(sal)
        elif method == 'gst':
            pos_indices = (sal >= val)
        elif method == 'random':
            pos_indices = np.random.permutation(len(sal))[:int(len(sal) * val / 100)]
            assert len(pos_indices) == int(len(sal) * val / 100)
        elif method == 'qa':
            pos_indices = ((node_type == 0) | (node_type == 1))[:len(sal)]
        elif method == "pos_only_binary":
            pos_indices = np.argwhere(sal > 0).reshape(-1)
        saliency_targets = np.zeros_like(sal)
        saliency_targets[pos_indices] = 1
        # assert not np.all(saliency_targets == 0)

    if np.all(saliency_targets == 0):
        saliency_targets[0] = 1
    assert not np.all(saliency_targets == 0)
    assert saliency_targets.shape == sal.shape
    return saliency_targets


def load_graph_data(adj_pk_path, num_choice):
    with open(adj_pk_path, 'rb') as fin:
        adj_concept_pairs = pickle.load(fin)
    n_samples = len(adj_concept_pairs)
    node_type_ids = torch.full((n_samples, 200), 2, dtype=torch.long)
    concept_ids = []
    for idx, (adj, concepts, qm, am) in tqdm.tqdm(enumerate(adj_concept_pairs), total=n_samples,
                                                  desc='loading adj matrices'):
        num_concept = min(len(concepts), 200)
        concepts = set(concepts[:num_concept])
        concept_ids.append(concepts)
        node_type_ids[idx, :num_concept][torch.tensor(qm, dtype=torch.bool)[:num_concept]] = 0
        node_type_ids[idx, :num_concept][torch.tensor(am, dtype=torch.bool)[:num_concept]] = 1

    node_type_ids = node_type_ids.view(-1, num_choice, *node_type_ids.size()[1:])

    return node_type_ids


def main(args):
    split_type = 'inhouse' if args.inhouse else 'official'
    if args.dataset == 'csqa' and not args.inhouse:
        split_type = "official"
    elif args.dataset == "csqa" and args.inhouse:
        split_type = 'inhouse'
    elif args.dataset == "obqa":
        split_type = 'official'
    elif args.dataset == "codah":
        split_type = "fold_0"
    elif args.dataset == "qasc":
        split_type = 'inhouse'

    if args.dataset == 'csqa' and split_type == 'inhouse':
        indices = np.arange(NUM_CSQA_INHOUSE[args.split])
    elif args.dataset == 'csqa' and split_type == 'official':
        indices = np.arange(NUM_CSQA_OFFICIAL[args.split])
    elif args.dataset == 'obqa' and split_type == 'official':
        indices = np.arange(NUM_OBQA_OFFICIAL[args.split])
    elif args.dataset == 'codah' and split_type == 'fold_0':
        indices = np.arange(NUM_CODAH_FOLD_0[args.split])
    elif args.dataset == 'qasc' and split_type == 'inhouse':
        indices = np.arange(NUM_QASC_INHOUSE[args.split])
    else:
        raise NotImplementedError

    if args.split == 'train' and args.train_percentage < 100:
        indices = indices[:int(len(indices) / 100 * args.train_percentage)]

    choice_keys = CHOICE_KEYS[:NUM_CHOICES[args.dataset]]

    # Create output dir
    bin_suffix = '' if args.train_percentage == 100 else str(args.train_percentage)
    output_dir = os.path.join(args.root_dir, args.dataset, split_type, args.text_encoder, 'bin' + bin_suffix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load saliency scores
    input_path = os.path.join('..', 'save', 'FS-{}'.format(args.exp), 'saliency',
                              'sal_fine_{}_{}_{}.csv'.format(args.saliency_method, args.saliency_source, args.split))
    assert os.path.exists(input_path)
    dataset_dict = load_sal_scores(input_path, args.saliency_method, indices, NUM_CHOICES[args.dataset])

    # type_ids_path = os.path.join(args.root_dir, 'mhgrn_data', args.dataset, 'graph', '{}.node_type_ids.pk'.format(args.split))
    # with open(type_ids_path, 'rb') as fin:
    #     node_type_ids = pickle.load(fin)

    if args.method == "raw":
        # use raw fine sal without binarization
        # since contains float, use pickle
        print("saving to tmp.pk")
        dataset_builders = defaultdict(list)
        filenames = {}
        for key in choice_keys:
            filename = os.path.join(output_dir, '{}.{}.sal_fine_{}_{}_{}_{}_{}_FS{}.pk'.format(
                args.split,
                args.graph_encoder,
                args.saliency_method,
                args.saliency_source,
                args.method,
                args.value,
                key,
                args.exp
            ))
            filenames[key] = filename

        for i in tqdm.tqdm(indices, total=len(indices), desc='Processing saliency scores',
                           bar_format=TRAINING_TQDM_BAD_FORMAT):
            for j, key in enumerate(filenames.keys()):
                saliency_scores = dataset_dict[i][j]
                dataset_builders[key].append(torch.from_numpy(saliency_scores))
        print(dataset_builders.keys())
        for key, filename in filenames.items():
            print("saving to ", filename)
            if Path(filename).exists(): os.unlink(Path(filename))
            torch.save(dataset_builders[key], filename)
    else:
        # Create indexed dataset builders
        dataset_builders, names = create_dataset(args, output_dir, choice_keys)

        # Build indexed datasets
        for i in tqdm.tqdm(indices, total=len(indices), desc='Processing saliency scores',
                           bar_format=TRAINING_TQDM_BAD_FORMAT):
            for j, key in enumerate(dataset_builders.keys()):
                saliency_scores = dataset_dict[i][j]
                # node_types = node_type_ids[i][j]
                if args.method == 'qa':
                    saliency_targets = get_target(args.method, saliency_scores, None, args.value)
                else:
                    saliency_targets = get_target(args.method, saliency_scores, None, args.value)
                dataset_builders[key].add_item(torch.from_numpy(saliency_targets))

        # Finalize indexed datasets
        finalize_dataset(dataset_builders, names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process saliency scores')
    parser.add_argument('-s', '--split', type=str, help='Dataset split', choices=['train', 'valid', 'test'],
                        required=True)
    parser.add_argument('-t', '--text-encoder', type=str,
                        choices=['roberta-base', 'roberta-large', 'albert-xxlarge-v2', 'bert-base-uncased'],
                        required=True)
    parser.add_argument('-g', '--graph-encoder', type=str, choices=['rn', 'mhgrn', 'pathgen'], required=True)
    parser.add_argument('--root-dir', type=str, default='../data/', help='dataset root directory')
    parser.add_argument('-d', '--dataset', type=str, choices=['csqa', 'obqa', 'codah', 'qasc'], required=True)
    parser.add_argument('--inhouse', action='store_true')
    parser.add_argument('-e', '--exp', type=int,
                        help='The experiment number that containing CSV file of saliency scores', required=True)
    parser.add_argument('--saliency-method', type=str, default='grad', choices=['grad', 'occl', 'ig'])
    parser.add_argument('--saliency-source', type=str, default='target', choices=['pred', 'target'])
    parser.add_argument('-m', '--method', type=str,
                        choices=['ratio', 'topk', 'gst', 'random', 'random_all', 'qa', 'pos_only_binary', 'raw'],
                        required=True)
    parser.add_argument('-v', '--value', type=float, required=True)
    parser.add_argument('--train-percentage', type=int, default=100)
    parser.add_argument(
        '--dataset-impl',
        metavar='FORMAT',
        default='mmap',
        choices=indexed_dataset.get_available_dataset_impl(),
        help='output dataset implementation')
    args = parser.parse_args()
    main(args)
