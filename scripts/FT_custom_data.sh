
# How to run this script
#bash scripts/FT_custom_data.sh

# Set the disease ID
# DISEASE_ID='MONDO_0004975'


# # Download and make the initial preparations
# python finetune_pinnacle/prepare_txdata.py --disease "$DISEASE_ID" \
# --result_dir data/therapeutic_target_task \
# --celltype_ppi data/networks/ppi_edgelists/ \
# --evidence_dir data/evidence_chembl/ \
# --all_drug_targets_path data/therapeutic_target_task/all_approved_oct2022.csv \
# --chembl2db_path data/src1src2.txt \

# # Generate the data splits
# python finetune_pinnacle/data_prep.py  \
#     --embeddings_dir data/pinnacle_embeds/ \
#     --embed pinnacle \
#     --disease "$DISEASE_ID" \
#     --result_dir data/therapeutic_target_task \
#     --embeddings_dir data/pinnacle_embeds/ \
#     --embed pinnacle \


DISEASE_IDS=('EFO_0003884' 'EFO_0003095' 'EFO_0003060' 'MONDO_0005184')  # Add more as needed

for DISEASE_ID in "${DISEASE_IDS[@]}"; do
    echo "Processing $DISEASE_ID"

    # Download and make the initial preparations
    python finetune_pinnacle/prepare_txdata.py --disease "$DISEASE_ID" \
        --result_dir data/therapeutic_target_task \
        --celltype_ppi data/networks/ppi_edgelists/ \
        --evidence_dir data/evidence_chembl/ \
        --all_drug_targets_path data/therapeutic_target_task/all_approved_oct2022.csv \
        --chembl2db_path data/src1src2.txt

    # Generate the data splits
    python finetune_pinnacle/data_prep.py \
        --embeddings_dir data/pinnacle_embeds/ \
        --embed pinnacle \
        --disease "$DISEASE_ID" \
        --result_dir data/therapeutic_target_task \
        --embeddings_dir data/pinnacle_embeds/ \
        --embed pinnacle
done
