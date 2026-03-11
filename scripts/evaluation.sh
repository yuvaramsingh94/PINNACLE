python evaluate/evaluate_target_prioritization.py --disease EFO_0000685 --test_only True \
--model_outputs_dir results/EFO_0000685_author_hpt/model_outputs/pinnacle_seed=1/ \
--evidence data/therapeutic_target_task/ \
--k 5 # this is for the APR@k