#!/bin/bash

for seed in 42
do
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port='29500' \
        --use_env main.py \
        imr_hideprompt_5e \
        --model vit_base_patch16_224 \
        --original_model vit_base_patch16_224 \
        --batch-size 24 \
        --epochs 50 \
        --data-path /data1/data \
        --lr 0.0005 \
        --ca_lr 0.005 \
        --crct_epochs 30 \
        --seed $seed \
        --train_inference_task_only \
        --output_dir ./output/imr_vit_multi_centroid_mlp_2_seed$seed
done


for seed in 42
do
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port='29510' \
        --use_env main.py \
        imr_hidelora \
        --model vit_base_patch16_224 \
        --original_model vit_base_patch16_224 \
        --batch-size 24 \
        --epochs 50 \
        --data-path /data1/data \
        --ca_lr 0.005 \
        --crct_epochs 30 \
        --seed 42 \
        --lr 0.03 \
        --lora_rank 5 \
        --reg 0.001 \
        --sched cosine \
        --dataset Split-Imagenet-R \
        --lora_momentum 0.1 \
        --lora_type hide \
        --trained_original_model ../../HiDe-Prompt/output/imr_vit_multi_centroid_mlp_2_seed42 \
        --output_dir ./output/imr_sup21k_lora_pe_seed42 \

done