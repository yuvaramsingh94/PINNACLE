#Code to sweep and find the best parameter
export CUDA_VISIBLE_DEVICES="7"
# (EFO_0000401)
python finetune_pinnacle/hyper_param_tune.py \
    --task_name EFO_0000685 \
    --embeddings_dir data/pinnacle_embeds/ \
    --positive_proteins_prefix data/therapeutic_target_task/EFO_0000685/positive_proteins_EFO_0000685 \
    --negative_proteins_prefix data/therapeutic_target_task/EFO_0000685/negative_proteins_EFO_0000685 \
    --data_split_path data/therapeutic_target_task/EFO_0000685/data_split_EFO_0000685 \
    --actn relu \
    --dropout 0.2 \
    --hidden_dim_1 32 \
    --hidden_dim_2 8 \
    --lr 0.01 \
    --norm bn \
    --order dn \
    --wd 0.001 \
    --random_state 1 \
    --num_epoch 50 \
    --batch_size 32