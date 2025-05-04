#DS
python -m pipeline.loop_regen_baseline --model deepseek-ai/deepseek-coder-6.7b-instruct --clf_model_path data_no_ignore/ds_cls.pt --file_path data_no_ignore/loop_gen_mbpp_ds.parquet --num_loops 3

python -m pipeline.loop_regen_org --model deepseek-ai/deepseek-coder-6.7b-instruct --clf_model_path data_no_ignore/ds_cls.pt --file_path data_no_ignore/loop_gen_mbpp_ds.parquet --num_loops 3

#CLM
python -m pipeline.loop_regen_baseline --model codellama/CodeLlama-7b-Instruct-hf --clf_model_path data_no_ignore/clm_cls.pt --file_path data_no_ignore/loop_gen_mbpp_clm.parquet --num_loops 3

python -m pipeline.loop_regen_org --model codellama/CodeLlama-7b-Instruct-hf --clf_model_path data_no_ignore/clm_cls.pt --file_path data_no_ignore/loop_gen_mbpp_clm.parquet --num_loops 3

#MGCD
python -m pipeline.loop_regen_baseline --model ise-uiuc/Magicoder-S-DS-6.7B --clf_model_path data_no_ignore/mgcd_cls.pt --file_path data_no_ignore/loop_gen_mbpp_mgcd.parquet --num_loops 3

python -m pipeline.loop_regen_org --model ise-uiuc/Magicoder-S-DS-6.7B --clf_model_path data_no_ignore/mgcd_cls.pt --file_path data_no_ignore/loop_gen_mbpp_mgcd.parquet --num_loops 3