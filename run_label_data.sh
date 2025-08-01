
python -m datalabel.label_mbpp_instruct_2 \
--data_root benchmark/MBPP/data \
--file intrinsic/LFCLF_2_embedding_mbpp_deepseek-ai_deepseek-coder-6.7b-instruct_32.parquet \
--model_name deepseek-ai/deepseek-coder-6.7b-instruct \
--lang python