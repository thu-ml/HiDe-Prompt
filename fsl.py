import datetime
from pathlib import Path
import sys
import time
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_upstream_continual_dataloader

import warnings
import argparse
import utils
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
import vits.hide_lora_vision_transformer
from engines import upstream_lora_engine, few_shot_engine


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def set_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

def split_continual_fs_datasets(args):
    args.continual_datasets_targets = []
    args.fs_datasets_targets = []
    for i in range(len(args.datasets)):
        cl_labels = [ i for i in range(args.continual_classes_per_dataset[i]) ]
        args.continual_datasets_targets.append(cl_labels)
        fs_labels = [ i for i in range(len(cl_labels), len(cl_labels) + args.few_shot_classes_per_dataset[i]) ]
        args.fs_datasets_targets.append(fs_labels)

def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    set_seed(args)
    split_continual_fs_datasets(args)

    data_loader, class_mask, target_dataset_map, target_task_map, task_dataset_map = build_upstream_continual_dataloader(args)

    print(class_mask)
    print(f"target dataset map: {target_dataset_map}")
    print(f"target task map: {target_task_map}")
    print(f"task dataset map: {task_dataset_map}")
    print(f"num datasets: {args.num_datasets}")
    print(f"num tasks: {args.num_tasks}")
    print(f"num classes: {args.nb_classes}")

    # vanilla model
    vanilla_model = create_model(args.model,
                                 pretrained=args.pretrained,
                                 num_classes=args.nb_classes,         
                                 )
    if args.freeze:
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in vanilla_model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False
    vanilla_model.to(device)

    # shared model
    shared_model = create_model(args.model,
                                pretrained=args.pretrained,
                                num_classes = args.nb_classes,
                                lora=True, 
                                lora_type='hide',
                                rank=args.lora_rank, 
                                lora_pool_size=args.num_datasets,                     
                                )
    if args.freeze:
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in shared_model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False
    shared_model.to(device)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0

    start_time = time.time()

    # train vanilla model
    if args.train_vanilla:
        model_without_ddp = vanilla_model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(vanilla_model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module
        n_parameters = sum(p.numel() for p in vanilla_model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        optimizer = create_optimizer(args, model_without_ddp)
        if args.sched != 'constant':
            lr_scheduler, _ = create_scheduler(args, optimizer)
        elif args.sched == 'constant':
            lr_scheduler = None
        criterion = torch.nn.CrossEntropyLoss().to(device)
        print(f"Start training for {args.epochs} epochs")
        upstream_lora_engine.train_and_evaluate_vanilla_model(vanilla_model, model_without_ddp, criterion, data_loader, optimizer, lr_scheduler, device, class_mask, target_dataset_map, target_task_map, task_dataset_map, args)


    # train shared model
    if args.train_shared:
        model_without_ddp = shared_model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(shared_model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module
        n_parameters = sum(p.numel() for p in shared_model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        base_params = [p for name, p in model_without_ddp.named_parameters() if 'lora' in name and p.requires_grad == True]
        base_fc_params = [p for name, p in model_without_ddp.named_parameters() if 'lora' not in name and p.requires_grad == True]
        base_params = {'params': base_params, 'lr': args.lr * 0.1, 'weight_decay': args.weight_decay}
        base_fc_params = {'params': base_fc_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
        network_params = [base_params, base_fc_params]
        optimizer = create_optimizer(args, network_params)
        if args.sched != 'constant':
            lr_scheduler, _ = create_scheduler(args, optimizer)
        elif args.sched == 'constant':
            lr_scheduler = None
        criterion = torch.nn.CrossEntropyLoss().to(device)
        print(f"Start training for {args.epochs} epochs")
        upstream_lora_engine.train_and_evaluate_shared_model(shared_model, model_without_ddp, vanilla_model, criterion, data_loader, optimizer, lr_scheduler, device, class_mask, target_dataset_map, target_task_map, task_dataset_map, args)

    if args.train_few_shot:
        # start to train and evalute few-shot datasets
        few_shot_dataset_idx = args.few_shot_dataset_idx
        class_idx = args.few_class_idx
        assert len(class_idx) == args.num_ways
        fs_model = create_model(args.model,
                                pretrained=args.pretrained,
                                num_classes=args.num_ways,         
                                )
        if args.freeze:
            # freeze args.freeze[blocks, patch_embed, cls_token] parameters
            for n, p in fs_model.named_parameters():
                if n.startswith(tuple(args.freeze)):
                    p.requires_grad = False

        fs_model.to(device)
        few_shot_engine.train_and_evaluate(vanilla_model, shared_model, fs_model, class_idx, device, target_dataset_map, args, i=0, dataset=args.datasets[few_shot_dataset_idx])


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Continual learning with LoRA configs")
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    #TODO: add config
    if config == 'imr_few_shot_lora':
        from configs.imr_few_shot_lora import get_args_parser
        config_parser = subparser.add_parser('imr_few_shot_lora', help='split-imagenetr lora config')
    elif config == 'cub_cars_few_shot_lora':
        from configs.cub_cars_few_shot_lora import get_args_parser
        config_parser = subparser.add_parser('cub_cars_few_shot_lora', help='split-cub and split-cars lora config')
    else:
        raise NotImplementedError
    get_args_parser(config_parser)
    args = parser.parse_args()
    print(args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.vanilla_model_output_dir:
        Path(args.vanilla_model_output_dir).mkdir(parents=True, exist_ok=True)
    if args.shared_model_output_dir:
        Path(args.shared_model_output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

