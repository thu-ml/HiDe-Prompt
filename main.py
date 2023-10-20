import os.path
import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_continual_dataloader

import utils
import warnings


warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')


def get_args():
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    config = parser.parse_known_args()[-1][0]
    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_hideprompt_5e':
        from configs.cifar100_hideprompt_5e import get_args_parser
        config_parser = subparser.add_parser('cifar100_hideprompt_5e', help='Split-CIFAR100 HiDe-Prompt configs')
    elif config == 'imr_hideprompt_5e':
        from configs.imr_hideprompt_5e import get_args_parser
        config_parser = subparser.add_parser('imr_hideprompt_5e', help='Split-ImageNet-R HiDe-Prompt configs')
    elif config == 'five_datasets_hideprompt_5e':
        from configs.five_datasets_hideprompt_5e import get_args_parser
        config_parser = subparser.add_parser('five_datasets_hideprompt_5e', help='five datasets HiDe-Prompt configs')
    elif config == 'cub_hideprompt_5e':
        from configs.cub_hideprompt_5e import get_args_parser
        config_parser = subparser.add_parser('cub_hideprompt_5e', help='Split-CUB HiDe-Prompt configs')
    elif config == 'cifar100_dualprompt':
        from configs.cifar100_dualprompt import get_args_parser
        config_parser = subparser.add_parser('cifar100_dualprompt', help='Split-CIFAR100 dual-prompt configs')
    elif config == 'imr_dualprompt':
        from configs.imr_dualprompt import get_args_parser
        config_parser = subparser.add_parser('imr_dualprompt', help='Split-ImageNet-R dual-prompt configs')
    elif config == 'five_datasets_dualprompt':
        from configs.five_datasets_dualprompt import get_args_parser
        config_parser = subparser.add_parser('five_datasets_dualprompt', help='five datasets dual-prompt configs')
    elif config == 'cub_dualprompt':
        from configs.cub_dualprompt import get_args_parser
        config_parser = subparser.add_parser('cub_dualprompt', help='Split-CUB dual-prompt configs')
    elif config == 'cifar100_sprompt_5e':
        from configs.cifar100_sprompt_5e import get_args_parser
        config_parser = subparser.add_parser('cifar100_sprompt_5e', help='Split-CIFAR100 s-prompt configs')
    elif config == 'imr_sprompt_5e':
        from configs.imr_sprompt_5e import get_args_parser
        config_parser = subparser.add_parser('imr_sprompt_5e', help='Split-ImageNet-R s-prompt configs')
    elif config == 'five_datasets_sprompt_5e':
        from configs.five_datasets_sprompt_5e import get_args_parser
        config_parser = subparser.add_parser('five_datasets_sprompt_5e', help='five datasets s-prompt configs')
    elif config == 'cub_sprompt_5e':
        from configs.cub_sprompt_5e import get_args_parser
        config_parser = subparser.add_parser('cub_sprompt_5e', help='Split-CUB s-prompt configs')
    elif config == 'cifar100_l2p':
        from configs.cifar100_l2p import get_args_parser
        config_parser = subparser.add_parser('cifar100_l2p', help='Split-CIFAR100 l2p configs')
    elif config == 'imr_l2p':
        from configs.imr_l2p import get_args_parser
        config_parser = subparser.add_parser('imr_l2p', help='Split-ImageNet-R l2p configs')
    elif config == 'five_datasets_l2p':
        from configs.five_datasets_l2p import get_args_parser
        config_parser = subparser.add_parser('five_datasets_l2p', help='five datasets l2p configs')
    elif config == 'cub_l2p':
        from configs.cub_l2p import get_args_parser
        config_parser = subparser.add_parser('cub_l2p', help='Split-CUB l2p configs')
    elif config == 'cifar100_hidelora':
        from configs.cifar100_hidelora import get_args_parser
        config_parser = subparser.add_parser('cifar100_hidelora', help='Split-CIFAR100 hidelora configs')
    elif config == 'imr_hidelora':
        from configs.imr_hidelora import get_args_parser
        config_parser = subparser.add_parser('imr_hidelora', help='Split-ImageNet-R hidelora configs')
    elif config == 'cifar100_continual_lora':
        from configs.cifar100_continual_lora import get_args_parser
        config_parser = subparser.add_parser('cifar100_continual_lora', help='Split-CIFAR100 continual lora configs')
    elif config == 'imr_continual_lora':
        from configs.imr_continual_lora import get_args_parser
        config_parser = subparser.add_parser('imr_continual_lora', help='Split-ImageNet-R continual lora configs')
    else:
        raise NotImplementedError

    get_args_parser(config_parser)
    args = parser.parse_args()
    args.config = config
    return args

def main(args):
    utils.init_distributed_mode(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if hasattr(args, 'train_inference_task_only') and args.train_inference_task_only:
        import trainers.tii_trainer as tii_trainer
        tii_trainer.train(args)
    elif 'hideprompt' in args.config and not args.train_inference_task_only:
        import trainers.hideprompt_trainer as hideprompt_trainer
        hideprompt_trainer.train(args)
    elif 'l2p' in args.config or 'dualprompt' in args.config or 'sprompt' in args.config:
        import trainers.dp_trainer as dp_trainer
        dp_trainer.train(args)
    elif 'hidelora' in args.config and not args.train_inference_task_only:
        import trainers.hidelora_trainer as hidelora_trainer
        hidelora_trainer.train(args)
    elif 'continual_lora' in args.config:
        import trainers.continual_lora_trainer as continual_lora_trainer
        continual_lora_trainer.train(args)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    
    args = get_args()
    print(args)
    main(args)
