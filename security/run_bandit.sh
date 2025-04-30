PATH_TO_OUTPUT_FILE="bandit_result_cl.json"
bandit -r source/LFCLF_embedding_security_codellama_CodeLlama-7b-Instruct-hf_1.parquet -f json -o $PATH_TO_OUTPUT_FILE
