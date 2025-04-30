DBNAME="cl70"
OUT="cl70.sarif"
FOLDER="LFCLF_embedding_security_codellama_CodeLlama-7b-Instruct-hf_1.parquet"
/root/QALLMCODE/WCODELLM/codeql/codeql/codeql database create $DBNAME  --language=python --source-root=source/${FOLDER}
/root/QALLMCODE/WCODELLM/codeql/codeql/codeql database analyze $DBNAME  --format=sarif-latest --output=$OUT
