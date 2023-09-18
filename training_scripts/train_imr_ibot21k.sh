#!/bin/bash

for seed in 42 40 44
do
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port='29501' \
        --use_env main.py \
        imr_hideprompt_5e \
        --model vit_base_patch16_224_21k_ibot \
        --original_model vit_base_patch16_224_21k_ibot \
        --batch-size 24 \
        --epochs 20 \
        --data-path ./datasets \
        --lr 0.0005 \
        --ca_lr 0.005 \
        --crct_epochs 30 \
        --seed $seed \
        --train_inference_task_only \
        --output_dir ./output/imr_ibot21k_multi_centroid_mlp_2_seed$seed
done

for seed in 42 40 44
do
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port='29502' \
        --use_env main.py \
        imr_hideprompt_5e \
        --model vit_base_patch16_224_21k_ibot \
        --original_model vit_base_patch16_224_21k_ibot \
        --batch-size 24 \
        --epochs 50 \
        --data-path ./datasets \
        --ca_lr 0.005 \
        --crct_epochs 30 \
	--sched constant \
        --seed $seed \
	--prompt_momentum 0.01 \
	--reg 0.5 \
	--length 20 \
        --larger_prompt_lr \
        --trained_original_model ./output/imr_ibot21k_multi_centroid_mlp_2_seed$seed \
	--output_dir ./output/imr_ibot21k_pe_seed$seed
done


