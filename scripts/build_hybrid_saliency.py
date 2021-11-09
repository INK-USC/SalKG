import os
import sys

sys.path.append(os.path.join(sys.path[0], '..'))
from helper import *


def main(args):
    assert (
            args.split != None
            and args.text_encoder != None
            and args.graph_encoder != None
            and args.target_type != None
    )
    if args.target_type == 'cls' or args.num_classes != None:
        assert args.target_type == 'cls' and args.num_classes >= 2
    inhouse = args.inhouse if args.dataset == 'csqa' else False
    if args.dataset == 'csqa' and not inhouse:
        raise NotImplementedError
    if args.dataset == "csqa":
        split_type = 'inhouse'
    elif args.dataset == "obqa":
        split_type = 'official'
    elif args.dataset == "codah":
        split_type = "fold_0"
    elif args.dataset == 'qasc':
        split_type = 'inhouse'
    num_choices = NUM_CHOICES[args.dataset]
    choice_keys = CHOICE_KEYS[:num_choices]

    if args.dataset == 'csqa' and split_type == 'inhouse':
        indices = np.arange(NUM_CSQA_INHOUSE[args.split])
    elif args.dataset == 'obqa' and split_type == 'official':
        indices = np.arange(NUM_OBQA_OFFICIAL[args.split])
    elif args.dataset == 'codah' and split_type == 'fold_0':
        indices = np.arange(NUM_CODAH_FOLD_0[args.split])
    elif args.dataset == 'qasc' and split_type == 'inhouse':
        indices = np.arange(NUM_QASC_INHOUSE[args.split])
    else:
        raise NotImplementedError

    if args.split == 'train' and inhouse and args.train_percentage < 100:
        indices = indices[:int(len(indices) / 100 * args.train_percentage)]

    # Create output dir
    if args.target_type == 'cls':
        bin_suffix = '' if args.train_percentage == 100 else str(args.train_percentage)
        output_dir = os.path.join(args.root_dir, args.dataset, split_type, args.text_encoder, 'bin' + bin_suffix)
        lp_output_dir = os.path.join(args.root_dir, args.dataset, split_type, args.text_encoder, 'npy')
        if not os.path.exists(lp_output_dir):
            os.makedirs(lp_output_dir)
    elif args.target_type == 'reg':
        output_dir = os.path.join(args.root_dir, args.dataset, split_type, args.text_encoder, 'json')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.saliency_method == 'qa':
        type_ids_path = os.path.join(args.root_dir, 'mhgrn_data', args.dataset, 'graph',
                                     '{}.node_type_ids.pk'.format(args.split))
        with open(type_ids_path, 'rb') as fin:
            node_type_ids = pickle.load(fin)

    if args.saliency_method == 'occl':
        no_kg_path = os.path.join(args.qa_no_kg_dir, f'sal_coarse_occl_target_{args.split}.csv')
        kg_path = os.path.join(args.qa_kg_dir, f'sal_fine_occl_pred_{args.split}.csv')
        targets, no_kg_preds = load_saliency(no_kg_path)
        targets_, kg_preds = load_saliency(kg_path)
        assert not np.any(targets - targets_)
        if args.saliency_method == 'qa':
            mask = np.zeros_like(kg_preds)
            node_type_sum = np.sum(node_type_ids.numpy().astype(int), axis=2)
            print(node_type_sum.shape)
            print(np.mean(node_type_sum))
            mask[node_type_sum > args.threshold] = 1
            sal = mask.astype(int)
        else:
            sal = compute_saliency(kg_preds - no_kg_preds, targets)
    else:
        sal_method = "occl" if args.saliency_method == "qa" else args.saliency_method
        sal_path = os.path.join(args.qa_kg_dir, f'sal_fine_{sal_method}_pred_{args.split}.csv')
        targets, raw_sal = load_saliency(sal_path, args.saliency_method)
        if args.saliency_method == 'qa':
            mask = np.zeros_like(raw_sal)
            print(mask.shape)
            print(node_type_ids.shape)
            node_type_sum = node_type_ids.sum(axis=2)
            print(mask.shape)
            print(node_type_sum.shape)
            mask[node_type_sum > args.threshold] = 1
            assert not (np.all(mask == 0) or np.all(mask == 1))
            sal = mask.astype(int)
        else:
            sal = compute_saliency(raw_sal, targets)

    # Initialize dataset
    if args.target_type == 'cls':
        if args.num_classes > 2:
            raise NotImplementedError
            if args.split == 'valid':
                valid_sal = sal
            else:
                valid_no_kg_path = os.path.join(args.qa_no_kg_dir, f'sal_coarse_occl_target_valid.csv')
                valid_kg_path = os.path.join(args.qa_kg_dir, f'sal_coarse_occl_target_valid.csv')
                valid_no_kg_preds = load_saliency(valid_no_kg_path)
                valid_kg_preds = load_saliency(valid_kg_path)
                valid_sal = compute_saliency(valid_no_kg_preds, valid_kg_preds)

            valid_bins = np.quantile(valid_sal, np.linspace(0, 1, args.num_classes + 1))
            valid_bins[0] = -float('inf')
            valid_bins[-1] = float('inf')

            sal_classes = np.digitize(sal, valid_bins) - 1

        # Create indexed dataset builders
        dataset_builders = {}
        for key in choice_keys:
            print(os.path.join(output_dir, '{}.{}.sal_hybrid_{}_target_{}{}_{}_FS{}_{}.bin'.format(
                args.split,
                args.graph_encoder,
                args.saliency_method,
                args.target_type,
                args.num_classes,
                key,
                args.no_kg_exp,
                args.kg_exp
            )))
            dataset_builders[key] = indexed_dataset.make_builder(
                os.path.join(output_dir, '{}.{}.sal_hybrid_{}_target_{}{}_{}_FS{}_{}.bin'.format(
                    args.split,
                    args.graph_encoder,
                    args.saliency_method,
                    args.target_type,
                    args.num_classes,
                    key,
                    args.no_kg_exp,
                    args.kg_exp
                )
                             ),
                impl=args.dataset_impl
            )
    elif args.target_type == 'reg':
        dataset = {}

    # Initialize progress bar
    pbar = tqdm.tqdm(
        total=len(indices),
        desc='Processing saliency scores',
        bar_format=TRAINING_TQDM_BAD_FORMAT,
    )

    # Build dataset
    if args.target_type == 'cls':
        sal_cls = []
    for i in indices:
        if args.target_type == 'cls' and args.num_classes == 2:
            for j, key in enumerate(dataset_builders.keys()):
                dataset_builders[key].add_item(torch.Tensor([sal[i, j]]))

        elif args.target_type == 'cls':
            raise NotImplementedError
            assert args.saliency_method == 'occl'
            for j, key in enumerate(dataset_builders.keys()):
                dataset_builders[key].add_item(torch.Tensor([sal_classes[i, j]]))
                sal_cls.append(sal_classes[i, j])
        else:
            raise NotImplementedError
            dataset[int(i)] = list(sal[i])

        pbar.update()

    pbar.close()

    if args.target_type == 'cls':
        # Finalize indexed datasets
        for key in choice_keys:
            dataset_builders[key].finalize(os.path.join(output_dir,
                                                        '{}.{}.sal_hybrid_{}_target_{}{}_{}_FS{}_{}.idx'.format(
                                                            args.split, args.graph_encoder, args.saliency_method,
                                                            args.target_type, args.num_classes, key, args.no_kg_exp,
                                                            args.kg_exp)))

        # # Save log prior
        # sal_cls = np.array(sal_cls)
        # lp = np.bincount(sal_cls, minlength=args.num_classes) / sal_cls.size
        # lp_path = os.path.join(lp_output_dir, '{}.{}.sal_coarse_occl_target_lp_{}{}.npy'.format(args.split, args.graph_encoder, args.target_type, args.num_classes))
        # np.save(lp_path, lp)

    elif args.target_type == 'reg':
        raise NotImplementedError
        output_file = os.path.join(output_dir,
                                   '{}.{}.sal_coarse_occl_target_{}.json'.format(args.split, args.graph_encoder,
                                                                                 args.target_type))
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)


def load_saliency(path, saliency_method="occl"):
    outputs = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row_arr = np.array([float(x) for x in row])
            outputs.append(row_arr[1:])
    outputs = np.stack(outputs)
    targets = outputs[:, 0].astype(int)
    sal = outputs[:, 1:]
    if saliency_method in ["occl", "goccl"]:
        assert np.all((np.sum(sal, axis=1) - np.ones(targets.shape)) < 1e-3)
        assert np.all(sal <= 1)
    return targets, sal


def compute_saliency(raw_sal, targets):
    if args.saliency_method == 'grad':
        mask = np.ones_like(raw_sal)
    else:
        mask = -1 * np.ones_like(raw_sal)
        mask[np.arange(len(raw_sal)), targets] = 1
    sal = (mask * raw_sal) > args.threshold
    return sal.astype(int)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process saliency scores')
    parser.add_argument('--split', type=str, help='Dataset split', choices=['train', 'valid', 'test'])
    parser.add_argument('--text-encoder', type=str,
                        choices=['roberta-base', 'roberta-large', 'albert-xxlarge-v2', 'bert-base-uncased'])
    parser.add_argument('--graph-encoder', type=str, choices=['rn', 'mhgrn', 'pathgen'])
    parser.add_argument('--root-dir', type=str, default='../data/', help='dataset root directory')
    parser.add_argument('--dataset', type=str, choices=['csqa', 'obqa', 'codah', 'qasc'])
    parser.add_argument('--inhouse', default=False, action='store_true')
    parser.add_argument('--saliency-method', type=str, default='occl', choices=['occl', 'grad', 'goccl', 'qa'])
    parser.add_argument('--qa-no-kg-dir', type=str,
                        help='Path to dir containing CSV file of non-KG model QA predictions')
    parser.add_argument('--qa-kg-dir', type=str, help='Path to dir containing CSV file of KG model QA predictions')
    parser.add_argument('--kg-exp', type=int, help='The experiment number that containing CSV file of saliency scores',
                        required=True)
    parser.add_argument('--no-kg-exp', type=int,
                        help='The experiment number that containing CSV file of saliency scores', required=True)
    parser.add_argument('--sal-dir', type=str, help='Path to dir containing CSV file of raw saliency scores')
    parser.add_argument('--target-type', type=str, choices=['cls', 'reg'])
    parser.add_argument('--num-classes', type=int)
    parser.add_argument('--threshold', type=float, default=0.01)
    parser.add_argument('--train-percentage', type=int, default=100)
    parser.add_argument(
        '--dataset-impl',
        metavar='FORMAT',
        default='mmap',
        choices=indexed_dataset.get_available_dataset_impl(),
        help='output dataset implementation')
    args = parser.parse_args()
    main(args)
