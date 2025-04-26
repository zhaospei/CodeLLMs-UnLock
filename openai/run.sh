python3 -m openai.create_batch  \
 --file openai/data/test_mbpp_ds.parquet\
 --output batch --dataset mbpp --model deepseek6.7 
python3 -m openai.create_batch  \
 --file openai/data/test_mbpp_clm.parquet\
 --output batch --dataset mbpp --model codellama7 
python3 -m openai.create_batch  \
 --file openai/data/test_mbpp_mgcd.parquet\
 --output batch --dataset mbpp --model magiccoder6.7 
#HUMAN EVAL
python3 -m openai.create_batch  \
 --file openai/data/test_human_eval_ds.parquet\
 --output batch --dataset human_eval --model deepseek6.7 
python3 -m openai.create_batch  \
 --file openai/data/test_human_eval_clm.parquet\
 --output batch --dataset human_eval --model codellama7 
python3 -m openai.create_batch  \
 --file openai/data/test_human_eval_mgcd.parquet\
 --output batch --dataset human_eval --model magiccoder6.7 

#DEV EVAL
python3 -m openai.create_batch  \
 --file openai/data/test_dev_eval_ds.parquet\
 --output batch --dataset dev_eval --model deepseek6.7 
python3 -m openai.create_batch  \
 --file openai/data/test_dev_eval_clm.parquet\
 --output batch --dataset dev_eval --model codellama7 
python3 -m openai.create_batch  \
 --file openai/data/test_dev_eval_mgcd.parquet\
 --output batch --dataset dev_eval --model magiccoder6.7 

