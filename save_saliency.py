import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import get_logger

logger = get_logger(__name__)


def restore_config_params(model, args):
    # restores some of the model args to those of config args
    model.args.saliency_mode = args.saliency_mode
    model.args.saliency_source = args.saliency_source
    model.args.saliency_method = args.saliency_method
    model.args.save_saliency = args.save_saliency
    model.args.ckpt_path = args.ckpt_path

    return model


def calc_fine_grad(args, batch, logits, sal_input) -> list:
    write_datas = []
    logits = logits.reshape(batch['size'], -1)
    num_choices = logits.shape[1]
    if args.graph_encoder in ['rn', 'pathgen']:
        num_nodes = sal_input[0].shape[1]
        num_tuples = batch['num_tuples'].reshape(batch['size'], num_choices)
        num_tuples += num_tuples == 0
        sal_input[1].retain_grad()
        for i in range(batch['size']):
            for j in range(num_choices):
                logits[i, j].backward(torch.ones_like(logits[i, j]), retain_graph=True)

                cur_concept_saliency_scores = torch.sum(
                    sal_input[0].view(batch['size'], num_choices, num_nodes, -1)[i, j]
                    * sal_input[0].grad.view(batch['size'], num_choices, num_nodes, -1)[i, j], dim=-1
                )[:num_tuples[i][j]].detach().cpu().numpy()

                cur_rel_saliency_scores = torch.sum(
                    sal_input[1].view(batch['size'], num_choices, num_nodes, -1)[i, j]
                    * sal_input[1].grad.view(batch['size'], num_choices, num_nodes, -1)[i, j], dim=-1
                )[:num_tuples[i][j]].detach().cpu().numpy()

                cur_saliency_scores = cur_concept_saliency_scores + cur_rel_saliency_scores

                # Normalize saliency scores
                cur_saliency_scores = cur_saliency_scores / np.linalg.norm(cur_saliency_scores)

                if batch['target'][i] != j:
                    cur_saliency_scores = -1 * cur_saliency_scores

                cur_index = batch['index'][i].item()
                cur_data = np.concatenate((np.array([cur_index, j]), cur_saliency_scores))
                write_datas.append(cur_data)
    else:
        # MHGRN
        num_nodes = sal_input.shape[1]
        adj_len = batch['adj_len'].reshape(batch['size'], num_choices)
        adj_len += adj_len == 0
        for i in range(batch['size']):
            for j in range(num_choices):
                logits[i, j].backward(torch.ones_like(logits[i, j]), retain_graph=True)

                cur_saliency_scores = torch.sum(
                    sal_input.view(batch['size'], num_choices, num_nodes, -1)[i, j]
                    * sal_input.grad.view(batch['size'], num_choices, num_nodes, -1)[i, j], dim=-1
                )[:adj_len[i][j]].detach().cpu().numpy()

                if batch['target'][i] != j:
                    cur_saliency_scores = -1 * cur_saliency_scores

                # Normalize saliency scores
                cur_saliency_scores = cur_saliency_scores / np.linalg.norm(cur_saliency_scores)

                cur_index = batch['index'][i].item()
                cur_data = np.concatenate((np.array([cur_index, j]), cur_saliency_scores))
                write_datas.append(cur_data)
    return write_datas


def save_saliency_scores(args, batch, logits, sal_input, saliency_path):
    """
    write saliency scores to `saliency_path`
    sal_input only in used if saliency_method = grad
    supports coarse occl, fine {occl, grad}
    """
    assert args.saliency_source == 'target'
    assert args.saliency_mode in ['coarse', 'fine']
    assert args.saliency_method in ['occl', 'grad']
    write_datas = []

    if args.saliency_mode == 'coarse':
        assert args.saliency_method == 'occl', "only support coarse occl"
        logits = logits.reshape(batch['size'], -1)
        probs = F.softmax(logits, dim=-1)
        for i in range(batch['size']):
            cur_index = batch['index'][i].item()
            cur_target = batch['target'][i].item()
            cur_pred = [x.item() for x in probs[i]]
            cur_data = [cur_index, cur_target] + cur_pred
            write_datas.append(cur_data)
    elif args.saliency_mode == 'fine':
        if args.saliency_method == 'occl':
            logits = logits.flatten()
            for i in range(batch['size']):
                cur_index = [batch['index'][i].item()]
                cur_fine_occl_id = [x.item() for x in batch['fine_occl_id'][i]]
                cur_target = [batch['target'][i].item()]
                cur_pred = [logits[i].item()]
                cur_data = cur_index + cur_fine_occl_id + cur_target + cur_pred
                write_datas.append(cur_data)
        elif args.saliency_method == 'grad':
            write_datas = calc_fine_grad(args, batch, logits, sal_input)
    with open(saliency_path, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        for cur_data in write_datas:
            writer.writerow(cur_data)


@torch.no_grad()
def save_saliency(dm, model, args):
    # xx/checkpoints/yy
    assert os.path.exists(args.ckpt_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # xx
    args.root_dir = Path(args.ckpt_path).parent.parent.resolve()
    # xx/saliency
    saliency_dir = os.path.join(args.root_dir, 'saliency')
    os.makedirs(saliency_dir, exist_ok=True)
    dataloaders_dict = {
        'train': dm.train_dataloader(shuffle=False),
        'valid': dm.val_dataloader(),
        'test': dm.test_dataloader(),
    }

    model = model.load_from_checkpoint(args.ckpt_path).to(device)
    # gnn need args to be set to save-saliency to output sal_input
    model = restore_config_params(model, args)

    for split, dataloader in dataloaders_dict.items():
        saliency_path = os.path.join(saliency_dir, 'sal_{}_{}_{}_{}.csv'.format(
            # fine or coarse
            args.saliency_mode,
            # occl or grad
            args.saliency_method,
            # from QA model or Sal model
            'target' if args.task == 'qa' else 'pred',
            # data split
            split
        )
                                     )
        # override if exist
        if os.path.exists(saliency_path):
            open(saliency_path, 'w').close()

        logger.info(f'Saving saliency at {saliency_path}.')

        with torch.set_grad_enabled(args.saliency_method == 'grad' and args.saliency_mode == 'fine'):
            for batch in tqdm(dataloader, total=len(dataloader)):
                """
                fine_occl_id (bsz, 3)
                mhgrn
                """
                batch = model.transfer_batch_to_device(batch)
                logits, sal_input = model(batch)
                save_saliency_scores(args, batch, logits, sal_input, saliency_path)
