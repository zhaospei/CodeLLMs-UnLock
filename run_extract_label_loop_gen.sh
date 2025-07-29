# #CLM  
# python -m datalabel.label_mbpp_instruct_2 \
# --data_root benchmark/MBPP/data \
# --file loop_gen/loop_gen_mbpp_clm_loop_5_base_full.parquet \
# --model_name codellama/CodeLlama-7b-Instruct-hf \
# --lang python

# python -m datalabel.label_mbpp_instruct_2 \
# --data_root benchmark/MBPP/data \
# --file loop_gen/loop_gen_mbpp_clm_loop_5_just_regen.parquet \
# --model_name codellama/CodeLlama-7b-Instruct-hf \
# --lang python

# #DS
# #CLM  
# python -m datalabel.label_mbpp_instruct_2 \
# --data_root benchmark/MBPP/data \
# --file loop_gen/loop_gen_mbpp_ds_loop_5_just_regen.parquet \
# --model_name deepseek-ai/deepseek-coder-6.7b-instruct \
# --lang python

# python -m datalabel.label_mbpp_instruct_2 \
# --data_root benchmark/MBPP/data \
# --file loop_gen/loop_gen_mbpp_ds_loop_5_base_full.parquet \
# --model_name deepseek-ai/deepseek-coder-6.7b-instruct \
# --lang python

# #MGCD

# python -m datalabel.label_mbpp_instruct_2 \
# --data_root benchmark/MBPP/data \
# --file loop_gen/loop_gen_mbpp_mgcd_loop_5_base_full.parquet \
# --model_name ise-uiuc/Magicoder-S-DS-6.7B \
# --lang python

# python -m datalabel.label_mbpp_instruct_2 \
# --data_root benchmark/MBPP/data \
# --file loop_gen/loop_gen_mbpp_mgcd_loop_5_just_regen.parquet \
# --model_name ise-uiuc/Magicoder-S-DS-6.7B \
# --lang python


python -m datalabel.label_mbpp_instruct_2 \
--data_root benchmark/MBPP/data \
--file sensitivity/LFCLF_embedding_mbpp_deepseek-ai_deepseek-coder-6.7b-instruct_32.parquet \
--model_name deepseek-ai/deepseek-coder-6.7b-instruct \
--lang python