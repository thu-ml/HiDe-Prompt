#!/bin/bash

for seed in 42
do
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port='29500' \
        --use_env main.py \
        cub_hideprompt_5e \
        --model vit_base_patch16_224 \
        --original_model vit_base_patch16_224 \
        --batch-size 24 \
        --epochs 20 \
        --data-path ./datasets \
        --lr 0.01 \
        --ca_lr 0.005 \
        --crct_epochs 30 \
        --seed $seed \
        --train_inference_task_only \
        --output_dir ./output/cub_vit_multi_centroid_mlp_2_seed$seed 
done



for seed in 42
do
python -m torch.distributed.launch \
	--nproc_per_node=8 \
	--master_port='29501' \
	--use_env main.py \
	cub_hideprompt_5e \
	--model vit_base_patch16_224 \
	--original_model vit_base_patch16_224 \
	--batch-size 24 \
	--epochs 50 \
	--data-path ./datasets \
	--ca_lr 0.005 \
	--crct_epochs 30 \
	--seed $seed \
	--prompt_momentum 0.01 \
	--reg 0.01 \
	--length 20 \
	--trained_original_model ./output/cub_vit_multi_centroid_mlp_2_seed$seed \
	--output_dir ./output/cub_vit_pe_seed$seed \
done
