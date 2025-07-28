PATH_TO_OUTPUT_FILE="bandit_result_cl13.json"
bandit -r source/LFCLF_embedding_security_codellama_CodeLlama-13b-Instruct-hf_1.parquet -f json -o $PATH_TO_OUTPUT_FILE
# ===
PATH_TO_OUTPUT_FILE="bandit_result_ds13.json"
bandit -r source/LFCLF_embedding_security_deepseek-ai_deepseek-coder-1.3b-instruct_1.parquet -f json -o $PATH_TO_OUTPUT_FILE
# ====
PATH_TO_OUTPUT_FILE="bandit_result_cg70.json"
bandit -r source/LFCLF_embedding_security_google_codegemma-7b-it_1.parquet -f json -o $PATH_TO_OUTPUT_FILE
