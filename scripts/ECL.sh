############################predict length 48####################################
python main_crossformer.py --data ECL \
--in_len 336 --out_len 48 --seg_len 6 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 5e-4  --lradj fixed --itr 5 

############################predict length 168####################################
python main_crossformer.py --data ECL \
--in_len 336 --out_len 168 --seg_len 12 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 5e-4  --lradj fixed --itr 5 

############################predict length 336####################################
python main_crossformer.py --data ECL \
--in_len 168 --out_len 336 --seg_len 24 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 5e-4  --lradj fixed --itr 5 

############################predict length 720####################################
python main_crossformer.py --data ECL \
--in_len 720 --out_len 720 --seg_len 24 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 5e-5  --lradj fixed  --itr 5 

############################predict length 960####################################
python main_crossformer.py --data ECL \
--in_len 720 --out_len 960 --seg_len 24 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 5e-5  --lradj fixed  --itr 5 

