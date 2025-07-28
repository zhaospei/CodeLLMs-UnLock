# SOURCE="/root/security/WCODELLM/output/codellama_CodeLlama-7b-Instruct-hf_human_eval_python_0_1_4_8_12_16_20_24_28_32/output2"
# python create_train_test_layer.py --label_file cl70.parquet --source_folder $SOURCE --outdir data/cl70
# SOURCE="/root/security/WCODELLM/output/ise-uiuc_Magicoder-S-DS-6.7B_human_eval_python_0_1_4_8_12_16_20_24_28_32/output2"
# python create_train_test_layer.py --label_file mc70.parquet --source_folder $SOURCE --outdir data/mc70
# SOURCE="/root/security/WCODELLM/output/deepseek-ai_deepseek-coder-6.7b-instruct_human_eval_python_0_1_4_8_12_16_20_24_28_32/output2"
# python create_train_test_layer.py --label_file ds67.parquet --source_folder $SOURCE --outdir data/ds67

SOURCE="/root/security/WCODELLM/output/deepseek-ai_deepseek-coder-1.3b-instruct_human_eval_python_0_1_4_8_12_16_20_24/output2"
python create_train_test_layer.py --label_file ds13.parquet --source_folder $SOURCE --outdir data/ds13
SOURCE="/root/security/WCODELLM/output/codellama_CodeLlama-13b-Instruct-hf_human_eval_python_0_1_4_8_12_16_20_24_28_32_36_40/output2"
python create_train_test_layer.py --label_file mc13.parquet --source_folder $SOURCE --outdir data/mc13
SOURCE="/root/security/WCODELLM/output/google_codegemma-7b-it_human_eval_python_0_1_4_8_12_16_20_24_28/output2"
python create_train_test_layer.py --label_file cg70.parquet --source_folder $SOURCE --outdir data/cg70