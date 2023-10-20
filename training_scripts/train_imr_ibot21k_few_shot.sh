python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port='29500' \
        --use_env fsl.py \
        imr_few_shot_lora \
        --model vit_base_patch16_224_21k_ibot \
        --original_model vit_base_patch16_224_21k_ibot \
        --batch-size 24 \
        --data-path ./datasets \
        --output_dir ./output/ \
        --seed 42 \
        --epochs 50 \
        --lr 0.01 \
        --lora_type hide \
        --num_tasks 8 \
        --fs_backbone vanilla \
        --train_vanilla \
        --vanilla_model_output_dir ./output/imr_ibot21k_vanilla_model \
        --shared_model_output_dir ./output/imr_ibot21k_shared_model \
        --lora_rank 8

python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port='29500' \
        --use_env fsl.py \
        imr_few_shot_lora \
        --model vit_base_patch16_224_21k_ibot \
        --original_model vit_base_patch16_224_21k_ibot \
        --batch-size 24 \
        --data-path ./datasets \
        --output_dir ./output/ \
        --seed 42 \
        --epochs 50 \
        --lr 0.01 \
        --lora_type hide \
        --num_tasks 8 \
        --fs_backbone shared \
        --train_shared \
        --vanilla_model_output_dir ./output/imr_ibot21k_vanilla_model \
        --shared_model_output_dir ./output/imr_ibot21k_shared_model \
        --lora_rank 8

python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port='29500' \
        --use_env fsl.py \
        imr_few_shot_lora \
        --model vit_base_patch16_224_21k_ibot \
        --original_model vit_base_patch16_224_21k_ibot \
        --batch-size 24 \
        --data-path ./datasets \
        --output_dir ./output/ \
        --seed 42 \
        --epochs 50 \
        --lr 0.01 \
        --lora_type hide \
        --num_tasks 8 \
        --num_fs_epochs 50 \
        --num_shots 5 \
        --train_few_shot \
        --fs_backbone shared \
        --num_episodes 50 \
        --vanilla_model_output_dir ./output/imr_ibot21k_vanilla_model \
        --shared_model_output_dir ./output/imr_ibot21k_shared_model \
        --lora_rank 8