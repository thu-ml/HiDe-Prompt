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
import tii_trainer
import warnings

warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    data_loader, data_loader_per_cls, class_mask, target_task_map = build_continual_dataloader(args)
    print(f"debug: {target_task_map}")

    if 'hide' in args.config:
        from engines.hide_wtp_and_tap_engine import train_and_evaluate, evaluate_till_now
        import vits.hide_vision_transformer as hide_vision_transformer
    elif 'dualprompt' in args.config or 'l2p' in args.config or 'sprompt' in args.config:
        from engines.dp_engine import train_and_evaluate, evaluate_till_now
        import vits.dp_vision_transformer as dp_vision_transformer
    else:
        raise NotImplementedError

    print(f"Creating original model: {args.original_model}")
    if 'hide' in args.config:
        original_model = create_model(
            args.original_model,
            pretrained=args.pretrained,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            mlp_structure=args.original_model_mlp_structure,
        )
    elif 'dualprompt' in args.config or 'l2p' in args.config or 'sprompt' in args.config:
        original_model = create_model(
            args.original_model,
            pretrained=args.pretrained,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
        )
    else:
        raise NotImplementedError

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        use_g_prompt=args.use_g_prompt,
        g_prompt_length=args.g_prompt_length,
        g_prompt_layer_idx=args.g_prompt_layer_idx,
        use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
        use_e_prompt=args.use_e_prompt,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
        same_key_value=args.same_key_value,
    )
    original_model.to(device)
    model.to(device)

    if args.freeze:
        # all backbobe parameters are frozen for original vit model
        for n, p in original_model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

    print(args)


    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            original_checkpoint_path = os.path.join(args.trained_original_model,
                                                    'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            if os.path.exists(original_checkpoint_path):
                print('Loading checkpoint from:', original_checkpoint_path)
                original_checkpoint = torch.load(original_checkpoint_path, map_location=device)
                original_model.load_state_dict(original_checkpoint['model'])
            else:
                print('No checkpoint found at:', original_checkpoint_path)
                return
            _ = evaluate_till_now(model, original_model, data_loader, device,
                                  task_id, class_mask, target_task_map, acc_matrix, args, )

        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0


    # This is a simple yet effective trick that helps to learn task-specific prompt better.
    if 'hide' in args.config and args.larger_prompt_lr:
        base_params = [p for name, p in model_without_ddp.named_parameters() if 'prompt' in name and p.requires_grad == True]
        base_fc_params = [p for name, p in model_without_ddp.named_parameters() if 'prompt' not in name and p.requires_grad == True]
        base_params = {'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
        base_fc_params = {'params': base_fc_params, 'lr': args.lr * 0.1, 'weight_decay': args.weight_decay}
        network_params = [base_params, base_fc_params]
        optimizer = create_optimizer(args, network_params)
    else:
        optimizer = create_optimizer(args, model_without_ddp)

    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model, model_without_ddp, original_model,
                       criterion, data_loader, data_loader_per_cls,
                       optimizer, lr_scheduler, device, class_mask, target_task_map, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")


if __name__ == '__main__':
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
    else:
        raise NotImplementedError

    get_args_parser(config_parser)

    args = parser.parse_args()

    args.config = config
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if 'hide' in config:
        if args.train_inference_task_only:
            tii_trainer.train_inference_task(args)
        else:
            main(args)
    else:
        main(args)
