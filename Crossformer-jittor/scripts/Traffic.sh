############################predict length 24####################################
python main_crossformer.py --data Traffic \
--in_len 96 --out_len 24 --seg_len 12 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 1e-3  --itr 5

############################predict length 48####################################
python main_crossformer.py --data Traffic \
--in_len 96 --out_len 48 --seg_len 12 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 1e-3 --itr 5

############################predict length 168####################################
python main_crossformer.py --data Traffic \
--in_len 336 --out_len 168 --seg_len 24 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 1e-3 --itr 5

############################predict length 336####################################
python main_crossformer.py --data Traffic \
--in_len 720 --out_len 336 --seg_len 24 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 1e-3 --itr 5

############################predict length 720####################################
python main_crossformer.py --data Traffic \
--in_len 336 --out_len 720 --seg_len 24 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 5e-3 --itr 5