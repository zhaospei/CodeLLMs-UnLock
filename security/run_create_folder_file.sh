FILE="/root/security/WCODELLM/output/codellama_CodeLlama-13b-Instruct-hf_human_eval_python_0_1_4_8_12_16_20_24_28_32_36_40/output2/LFCLF_embedding_security_codellama_CodeLlama-13b-Instruct-hf_1.parquet"
OUTDIR="source/LFCLF_embedding_security_codellama_CodeLlama-13b-Instruct-hf_1.parquet"
python create_folder_file.py --file $FILE --out_dir $OUTDIR
# ===
FILE_NAME='LFCLF_embedding_security_deepseek-ai_deepseek-coder-1.3b-instruct_1.parquet'
FILE="/root/security/WCODELLM/output/deepseek-ai_deepseek-coder-1.3b-instruct_human_eval_python_0_1_4_8_12_16_20_24/output2/$FILE_NAME"
OUTDIR="source/$FILE_NAME"
python create_folder_file.py --file $FILE --out_dir $OUTDIR
# ===
FILE_NAME='LFCLF_embedding_security_google_codegemma-7b-it_1.parquet'
FILE="/root/security/WCODELLM/output/google_codegemma-7b-it_human_eval_python_0_1_4_8_12_16_20_24_28/output2/$FILE_NAME"
OUTDIR="source/$FILE_NAME"
python create_folder_file.py --file $FILE --out_dir $OUTDIR