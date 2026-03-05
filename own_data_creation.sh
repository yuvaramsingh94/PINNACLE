
#cd finetune_pinnacle
pwd
python finetune_pinnacle/prepare_txdata.py --disease 'EFO_0004997' \
--celltype_ppi /storage/nas/da24s015/PINNACLE/data/networks/ppi_edgelists/ \
--all_drug_targets_path /storage/nas/da24s015/PINNACLE/data/therapeutic_target_task/all_approved_oct2022.csv \
--chembl2db_path /storage/nas/da24s015/PINNACLE/data/src1src2.txt \
--evidence_dir /storage/nas/da24s015/PINNACLE/data/evidence_chembl   \