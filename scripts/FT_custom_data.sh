
# How to run this script
#bash scripts/FT_custom_data.sh

# Download and make the initial preparations
python finetune_pinnacle/prepare_txdata.py --disease 'EFO_0000401' \
--result_dir data/therapeutic_target_task \
--celltype_ppi data/networks/ppi_edgelists/ \
--evidence_dir data/evidence_chembl/ \
--all_drug_targets_path data/therapeutic_target_task/all_approved_oct2022.csv \
--chembl2db_path data/src1src2.txt \

# Generate the data splits
python finetune_pinnacle/data_prep.py  \
    --embeddings_dir data/pinnacle_embeds/ \
    --embed pinnacle \
    --disease 'EFO_0000401' \
    --result_dir data/therapeutic_target_task \
    --embeddings_dir data/pinnacle_embeds/ \
    --embed pinnacle \
