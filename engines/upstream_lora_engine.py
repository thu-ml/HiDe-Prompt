import datetime
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable
import numpy as np
import torch
from timm.optim import create_optimizer
import utils
from timm.utils import accuracy

def train_and_evaluate_vanilla_model(vanilla_model, model_without_ddp, criterion, data_loader, optimizer, lr_scheduler, device, class_mask, target_dataset_map, target_task_map, task_dataset_map, args):
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    for task_id in range(args.num_tasks):
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model_without_ddp)
        
        for epoch in range(args.epochs): 
            train_stats = train_one_epoch(model=vanilla_model, criterion=criterion, data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                          device=device, epoch=epoch, task_dataset_map=task_dataset_map, max_norm=args.clip_grad, set_training_mode=True, task_id=task_id, 
                                          class_mask=class_mask, args = args, training_type='vanilla')
    
            if lr_scheduler:
                lr_scheduler.step(epoch)
        
        if args.vanilla_model_output_dir and utils.is_main_process():
            Path(os.path.join(args.vanilla_model_output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.vanilla_model_output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

        test_stats = evaluate_till_now(model=vanilla_model, data_loader=data_loader,
                                       device=device, task_id=task_id, class_mask=class_mask,
                                       acc_matrix=acc_matrix, args=args, 
                                       evaluate_type='vanilla', original_model=vanilla_model,
                                       target_dataset_map=target_dataset_map, 
                                       target_task_map=target_task_map, task_dataset_map=task_dataset_map)


def train_and_evaluate_shared_model(shared_model, model_without_ddp, vanilla_model, criterion, data_loader, optimizer, lr_scheduler, device, class_mask, target_dataset_map, target_task_map, task_dataset_map, args):
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    for task_id in range(args.num_tasks):
        if task_id > 0 and args.reinit_optimizer:
            base_params = [p for name, p in model_without_ddp.named_parameters() if 'lora' in name and p.requires_grad == True]
            base_fc_params = [p for name, p in model_without_ddp.named_parameters() if 'lora' not in name and p.requires_grad == True]
            base_params = {'params': base_params, 'lr': args.lr * 0.1, 'weight_decay': args.weight_decay}
            base_fc_params = {'params': base_fc_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
            network_params = [base_params, base_fc_params]
            optimizer = create_optimizer(args, network_params)
        
        original_checkpoint_path = os.path.join(args.vanilla_model_output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
        if os.path.exists(original_checkpoint_path):
                print('Loading checkpoint from:', original_checkpoint_path)
                original_checkpoint = torch.load(original_checkpoint_path, map_location=device)
                vanilla_model.load_state_dict(original_checkpoint['model'], strict=True)
        else:
            print('No checkpoint found at:', original_checkpoint_path)
            return
        
        for epoch in range(args.epochs): 
            train_stats = train_one_epoch(model=shared_model, criterion=criterion, data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                          device=device, epoch=epoch, task_dataset_map=task_dataset_map, max_norm=args.clip_grad, set_training_mode=True, task_id=task_id, 
                                          class_mask=class_mask, args = args, training_type='shared')
    
            if lr_scheduler:
                lr_scheduler.step(epoch)
        
        if args.shared_model_output_dir and utils.is_main_process():
            Path(os.path.join(args.shared_model_output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.shared_model_output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

        test_stats = evaluate_till_now(model=shared_model, data_loader=data_loader,
                                       device=device, task_id=task_id, class_mask=class_mask,
                                       acc_matrix=acc_matrix, args=args, 
                                       evaluate_type='shared', original_model=vanilla_model,
                                       target_dataset_map=target_dataset_map, 
                                       target_task_map=target_task_map, task_dataset_map=task_dataset_map)


def train_one_epoch(model: torch.nn.Module, criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, task_dataset_map: dict, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args=None, training_type='vanilla'):
    model.train(set_training_mode)
    
    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)
        
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if training_type == 'shared':
            output = model(input, task_id=task_dataset_map[task_id], train=set_training_mode)
        else:
            output = model(input, task_id=task_id, train=set_training_mode)
        logits = output['logits']

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target)  # base criterion (CrossEntropyLoss)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
             device, i=-1, task_id=-1, class_mask=None, target_task_map=None, args=None, evaluate_type='vanilla',
             target_dataset_map=None, task_dataset_map=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(i + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if evaluate_type == 'vanilla':
                output = model(input)
                logits = output['logits']
                if args.train_mask and class_mask is not None:
                    mask = []
                    for id in range(task_id + 1):
                        mask.extend(class_mask[id])
                    not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                    not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                    logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                lora_id = torch.max(logits, dim=1)[1] 
                lora_id = torch.tensor([target_dataset_map[v.item()] for v in lora_id], device=device)
                task_inference_acc = utils.task_inference_accuracy(lora_id.unsqueeze(-1), target, target_dataset_map)
            
            elif evaluate_type == 'shared':
                with torch.no_grad():
                    if original_model is not None:
                        output = original_model(input)
                        logits = output['logits']
                        if args.train_mask and class_mask is not None:
                            mask = []
                            for id in range(task_id + 1):
                                mask.extend(class_mask[id])
                            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                        lora_id = torch.max(logits, dim=1)[1]
                        lora_id = torch.tensor([target_dataset_map[v.item()] for v in lora_id], device=device)
                        # print(lora_id)
                        task_inference_acc = utils.task_inference_accuracy(lora_id.unsqueeze(-1), target, target_dataset_map)
                    else:
                        raise NotImplementedError("original model is None")

                    output = model(input, task_id=lora_id)
                    logits = output['logits']

            else:
                with torch.no_grad():
                    if original_model is not None:
                        output = original_model(input)
                        logits = output['logits']
                        if args.train_mask and class_mask is not None:
                            mask = []
                            for id in range(task_id + 1):
                                mask.extend(class_mask[id])
                            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                        lora_id = torch.max(logits, dim=1)[1]
                        lora_id = torch.tensor([target_task_map[v.item()] for v in lora_id], device=device)
                        # print(lora_id)
                        task_inference_acc = utils.task_inference_accuracy(lora_id.unsqueeze(-1), target, target_task_map)
                    else:
                        raise NotImplementedError("original model is None")

                    output = model(input, task_id=lora_id)
                    logits = output['logits']


            if args.task_inc and class_mask is not None:
                # adding mask to output logits
                mask = class_mask[i]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
            metric_logger.meters['Acc@task'].update(task_inference_acc.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@task {task_acc.global_avg:.3f} Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(task_acc=metric_logger.meters['Acc@task'],
                top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
                      device, task_id=-1, class_mask=None, target_task_map=None, 
                      target_dataset_map=None, task_dataset_map=None,
                      acc_matrix=None, args=None, evaluate_type='vanilla'):
    stat_matrix = np.zeros((4, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id + 1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'],
                              device=device, i=i, task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                              args=args, evaluate_type=evaluate_type, target_dataset_map=target_dataset_map, task_dataset_map=task_dataset_map)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']
        stat_matrix[3, i] = test_stats['Acc@task']

        acc_matrix[i, task_id] = test_stats['Acc@1']

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@task: {:.4f}\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
        task_id + 1,
        avg_stat[3],
        avg_stat[0],
        avg_stat[1],
        avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats


