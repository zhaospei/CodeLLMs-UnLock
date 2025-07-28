DBNAME="cl13"
OUT="cl13.sarif"
FOLDER="LFCLF_embedding_security_codellama_CodeLlama-13b-Instruct-hf_1.parquet"
/root/QALLMCODE/WCODELLM/codeql/codeql/codeql database create $DBNAME  --language=python --source-root=source/${FOLDER}
/root/QALLMCODE/WCODELLM/codeql/codeql/codeql database analyze $DBNAME  --format=sarif-latest --output=$OUT
# ====
DBNAME="ds13"
OUT="ds13.sarif"
FOLDER="LFCLF_embedding_security_deepseek-ai_deepseek-coder-1.3b-instruct_1.parquet"
/root/QALLMCODE/WCODELLM/codeql/codeql/codeql database create $DBNAME  --language=python --source-root=source/${FOLDER}
/root/QALLMCODE/WCODELLM/codeql/codeql/codeql database analyze $DBNAME  --format=sarif-latest --output=$OUT
# ====
DBNAME="cg70"
OUT="cg70.sarif"
FOLDER="LFCLF_embedding_security_google_codegemma-7b-it_1.parquet"
/root/QALLMCODE/WCODELLM/codeql/codeql/codeql database create $DBNAME  --language=python --source-root=source/${FOLDER}
/root/QALLMCODE/WCODELLM/codeql/codeql/codeql database analyze $DBNAME  --format=sarif-latest --output=$OUT
