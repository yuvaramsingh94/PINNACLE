#Code to sweep and find the best parameter
export CUDA_VISIBLE_DEVICES="7"
# (EFO_0000401)
python finetune_pinnacle/hyper_param_tune.py \
    --task_name MONDO_0005184 \
    --embeddings_dir data/pinnacle_embeds/ \
    --positive_proteins_prefix data/therapeutic_target_task/MONDO_0005184/positive_proteins_MONDO_0005184 \
    --negative_proteins_prefix data/therapeutic_target_task/MONDO_0005184/negative_proteins_MONDO_0005184 \
    --data_split_path data/therapeutic_target_task/MONDO_0005184/data_split_MONDO_0005184 \
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
    --batch_size 32 \
    --lr_scheduler  ## TODO Check this if it helps