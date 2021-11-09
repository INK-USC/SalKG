import argparse
import os
import socket
import sys
import time
import uuid
from argparse import ArgumentParser
from configparser import ConfigParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from data import QADataModule, SaliencyDataModule
from qa_model import QAModel
from saliency_model import SaliencyModel
from save_saliency import save_saliency
from utils import get_logger

model_dict = {
    'qa': (QAModel, QADataModule),
    'saliency': (SaliencyModel, SaliencyDataModule)
}

monitor_dict = {
    'qa': 'valid_acc_epoch',
    'saliency': {
        'cls': ['valid_qa_acc_epoch', 'valid_f1_epoch', 'valid_acc_epoch'],
    }
}
neptune_api_key = os.environ['NEPTUNE_API_TOKEN']
neptune_project_name = os.environ['NEPTUNE_PROJ_NAME']
basic_logger = get_logger(__name__)


def parse_args():
    conf_parser = ArgumentParser()
    conf_parser.add_argument('--config')
    args, _ = conf_parser.parse_known_args()

    # check if config path is valid
    if not os.path.isfile(args.config):
        print('Config path invalid')
        sys.exit(0)

    config = ConfigParser()
    config.read([args.config])
    defaults = dict(config.items('DEFAULT'))
    for k, v in defaults.items():
        if v == 'True':
            defaults[k] = True
        elif v == 'False':
            defaults[k] = False
        elif v == 'None':
            defaults[k] = None

    parser = argparse.ArgumentParser(parents=[conf_parser], add_help=False)
    parser = pl.Trainer.add_argparse_args(parser)
    model, dm = model_dict[defaults['task']]
    parser = model.add_model_specific_args(parser)
    parser = dm.add_data_specific_args(parser)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--freeze_epochs', type=int, default=-1)
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--tag_attrs', type=str, default='task,dataset,arch,graph_encoder')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_checkpoint', action='store_true')
    parser.set_defaults(**defaults)

    return parser.parse_args()


def get_neptune_logger(args):
    tags = []
    args_dict = vars(args)
    args_dict['hostname'] = socket.gethostname()
    for tag_attr in args.tag_attrs.split(','):
        if args_dict.get(tag_attr, None) is not None:
            tags.append(args_dict[tag_attr])

    neptune_logger = NeptuneLogger(
        api_key=neptune_api_key,
        project_name=neptune_project_name,
        experiment_name=args.name,
        params=args_dict,
        tags=tags,
        offline_mode=args.debug,
    )
    # new version of neptune logger will not create experiment in init
    neptune_logger.experiment

    return neptune_logger


def get_callbacks(args):
    if args.task == 'qa':
        monitor = monitor_dict[args.task]
        mode = 'max'
    elif args.task == 'saliency':
        monitor = monitor_dict[args.task][args.target_type]
        if args.target_type == 'cls':
            if args.pruned_qa:
                monitor = monitor[0]
            elif args.sal_num_classes == 2:
                monitor = monitor[1]
            elif args.sal_num_classes > 2:
                monitor = monitor[2]
        mode = 'max' if args.target_type == 'cls' else 'min'

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=os.path.join(args.root_dir, 'checkpoints'),
        save_top_k=1,
        mode=mode,
        verbose=True,
        save_last=False,
    )

    early_stop_callback = EarlyStopping(
        monitor=monitor,
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode=mode
    )

    return [checkpoint_callback, early_stop_callback]


def build(args):
    """
    build pl modules
    """
    pl.seed_everything(args.seed)
    basic_logger.info('Loading data...')
    model, dm = model_dict[args.task]
    dm = dm(args)
    dm.setup()
    basic_logger.info('Loading model...')
    model = model(args)

    return dm, model


def train(dm, model, args):
    neptune_logger = get_neptune_logger(args)
    args.root_dir = f'../save2/{neptune_logger.experiment_id}'

    basic_logger.info('Building trainer...')
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=neptune_logger,
        callbacks=get_callbacks(args),
        precision=16 if args.fp16 else 32,
        num_sanity_val_steps=0
    )
    trainer.fit(model, dm)
    basic_logger.info('Testing the best model...')
    trainer.test(ckpt_path='best')


if __name__ == '__main__':
    args = parse_args()
    basic_logger.info("initializing pl modules")
    args.name = f'{args.task}_{args.arch}_{time.strftime("%d_%m_%Y")}_{time.strftime("%H:%M:%S")}_{str(uuid.uuid4())[:8]}'
    dm, model = build(args)

    if args.save_saliency:
        save_saliency(dm, model, args)
    else:
        train(dm, model, args)
