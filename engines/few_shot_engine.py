from few_shot_datasets import get_episode_dataset, get_query_dataset, EpisodeSampler
from datasets import get_dataset, build_cifar_transform, build_transform
import random
import torch
from timm.utils import accuracy
import numpy as np
import os
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

def train_and_evaluate(vanilla_model, shared_model, model, class_ids, device, target_dataset_map, args, i, dataset):
        # Load data
        if 'cifar' in dataset.lower():
            transform_train = build_cifar_transform(True, args)
            transform_val = build_cifar_transform(False, args)
        else:
            transform_train = build_transform(True, args)
            transform_val = build_transform(False, args)
        dataset_train, dataset_val = get_dataset(dataset.replace('Split-', ''), transform_train, transform_val, args)
        episode_samples = EpisodeSampler(data_source=dataset_train, class_ids=class_ids, num_episodes=args.num_episodes, num_ways=args.num_ways, num_shots=args.num_shots)
        dataloader_val_sample = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=4, prefetch_factor=100)
        query_sampels_dict = defaultdict(list)
        for j, (_, label) in enumerate(dataloader_val_sample):
                if label in class_ids:
                    query_sampels_dict[label.item()].append(j)
        acc = []
        print('start few-shot learning')
        for k, episode_sample in enumerate(episode_samples):
            print(f'No.{k} episode samples: {episode_sample}')
            episode_dataset = get_episode_dataset(dataset=dataset.replace('Split-', ''), episode_samples=episode_sample, data_source=dataset_train)
            
            query_sampels = []
            
            for c in class_ids:
                selected_samples = random.sample(query_sampels_dict[c], min(args.num_queries, len(query_sampels_dict[c])))
                query_sampels.extend(selected_samples)
            print(f'No.{k} query samples {query_sampels}')
            query_dataset = get_query_dataset(dataset=dataset.replace('Split-', ''), query_samples=query_sampels, data_source=dataset_val)
            # dataloader
            dataloader_train = torch.utils.data.DataLoader(episode_dataset, batch_size=args.num_shots * args.num_ways, shuffle=True, num_workers=args.num_workers)
            dataloader_val = torch.utils.data.DataLoader(query_dataset, batch_size=args.num_queries, shuffle=False, num_workers=args.num_workers)
            
            # load checkpoint
            if args.fs_backbone == 'shared':
                shared_model_state_dict = torch.load(os.path.join(args.shared_model_output_dir, f'checkpoint/task{args.tasks_per_dataset[i]}_checkpoint.pth'), map_location=device)['model']
                shared_model.load_state_dict(shared_model_state_dict, strict=True)
                shared_model.update_attention(task_id=i, device=device)
                shared_model_state_dict = shared_model.state_dict()
                state_dict_without_head = {}
                for key,val in shared_model_state_dict.items():
                    if 'head' not in key:
                        state_dict_without_head[key] = val
                model.load_state_dict(state_dict_without_head, strict=False)
            if args.fs_backbone == 'vanilla':
                vanilla_model_state_dict = torch.load(os.path.join(args.vanilla_model_output_dir, f'checkpoint/task{args.tasks_per_dataset[i]}_checkpoint.pth'), map_location=device)['model']
                vanilla_model.load_state_dict(vanilla_model_state_dict, strict=True)
                vanilla_model_state_dict = vanilla_model.state_dict()
                state_dict_without_head = {}
                for key,val in vanilla_model_state_dict.items():
                    if 'head' not in key:
                        state_dict_without_head[key] = val
                model.load_state_dict(state_dict_without_head, strict=False)
            if args.fs_backbone == 'vanilla+shared':
                shared_model_state_dict = torch.load(os.path.join(args.shared_model_output_dir, f'checkpoint/task{args.tasks_per_dataset[i]}_checkpoint.pth'), map_location=device)['model']
                shared_model.load_state_dict(shared_model_state_dict, strict=True)
                vanilla_model_state_dict = torch.load(os.path.join(args.vanilla_model_output_dir, f'checkpoint/task{args.tasks_per_dataset[i]}_checkpoint.pth'), map_location=device)['model']
                vanilla_model.load_state_dict(vanilla_model_state_dict, strict=True)
                for input, target in dataloader_train:
                    input = input.to(device, non_blocking=True)
                    output = vanilla_model(input)
                    logits = output["logits"]
                    lora_id = torch.max(logits, dim=1)[1] 
                    lora_id = torch.tensor([target_dataset_map[v.item()] for v in lora_id], device=device)
                    lora_id = torch.argmax(torch.bincount(lora_id))
                    print(lora_id)
                shared_model.update_attention(task_id=lora_id, device=device)
                shared_model_state_dict = shared_model.state_dict()
                state_dict_without_head = {}
                for key,val in shared_model_state_dict.items():
                    if 'head' not in key:
                        state_dict_without_head[key] = val
                model.load_state_dict(state_dict_without_head, strict=False)
            model.reset_classifier()
            model.to(device)

            # optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=args.fs_lr)
            # loss
            criterion = torch.nn.CrossEntropyLoss()

            # train
            label_encoder = LabelEncoder()
            label_encoder.fit(class_ids)
            model.train()
            for epoch in range(args.num_fs_epochs):
                for input, target in dataloader_train:
                    input = input.to(device, non_blocking=True)
                    target = torch.tensor(label_encoder.transform(target)).to(device, non_blocking=True)
                    optimizer.zero_grad()
                    output = model(input)
                    loss = criterion(output['logits'], target)
                    loss.backward()
                    optimizer.step()
                print('epoch: {}, loss: {}'.format(epoch, loss.item()))

            model.eval()
            # evaluate
            acc1 = 0
            with torch.no_grad():
                for input, target in dataloader_val:
                    input = input.to(device, non_blocking=True)
                    target = torch.tensor(label_encoder.transform(target)).to(device, non_blocking=True)
                    output = model(input)
                    logits = output['logits']
                    loss = criterion(logits, target)
                    _, pred = torch.max(logits, 1)
                    acc1 += torch.sum(pred == target) 
                    
                    print('val loss: {}'.format(loss.item()))
                print('acc: {}'.format(acc1.item() / len(dataloader_val.dataset)))
            
            acc.append(acc1.item() / len(dataloader_val.dataset))
                
        # average result    
        mean = np.mean(np.array(acc))
        var = np.var(np.array(acc))
        print('mean acc: {}, var acc: {}'.format(mean, var))
        print('95 confidence interval {}'.format(1.96 * np.sqrt(var)))

