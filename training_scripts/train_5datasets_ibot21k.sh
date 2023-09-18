#!/bin/bash

for seed in 42 40 44
do
python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --use_env main.py \
        five_datasets_hideprompt_5e \
        --original_model vit_base_patch16_224_21k_ibot \
        --model vit_base_patch16_224_21k_ibot \
        --batch-size 32 \
        --data-path ./datasets \
        --output_dir ./output/5datasets_ibot21k_multi_centroid_mlp_2_seed$seed \
        --epochs 20 \
        --sched constant \
        --seed $seed \
        --train_inference_task_only \
        --lr 0.001 \
done

for seed in 42 40 44
do
python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --use_env main.py \
        five_datasets_hideprompt_5e \
        --original_model vit_base_patch16_224_21k_ibot \
        --model vit_base_patch16_224_21k_ibot \
        --batch-size 32 \
        --data-path ./datasets \
        --output_dir ./output/5datasets_ibot21k_pe_seed$seed \
        --epochs 20 \
        --sched constant \
        --lr 0.03 \
        --clip-grad 2 \
        --reg 0.1 \
        --prompt_momentum 0.01 \
        --seed $seed \
        --larger_prompt_lr \
        --trained_original_model ./output/5datasets_ibot21k_multi_centroid_mlp_2_seed$seed
done