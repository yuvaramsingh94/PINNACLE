#cd finetune_pinnacle
export CUDA_VISIBLE_DEVICES="7"
# (EFO_0000401)
python finetune_pinnacle/train.py \
    --task_name EFO_0003884 \
    --embeddings_dir data/pinnacle_embeds/ \
    --positive_proteins_prefix data/therapeutic_target_task/EFO_0003884/positive_proteins_EFO_0003884 \
    --negative_proteins_prefix data/therapeutic_target_task/EFO_0003884/negative_proteins_EFO_0003884 \
    --data_split_path data/therapeutic_target_task/EFO_0003884/data_split_EFO_0003884 \
    --actn relu \
    --dropout 0.2 \
    --hidden_dim_1 8 \
    --hidden_dim_2 16 \
    --lr 0.01 \
    --norm ln \
    --order dn \
    --wd 0.001 \
    --random_state 1 \
    --num_epoch 150 \
    --batch_size 32 \
    --lr_scheduler