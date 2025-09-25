############################predict length 24####################################
python main_crossformer.py --data ETTm1 \
--in_len 288 --out_len 24 --seg_len 12 \
--learning_rate 1e-4 --itr 5 

############################predict length 48####################################
python main_crossformer.py --data ETTm1 \
--in_len 288 --out_len 48 --seg_len 6 \
--learning_rate 1e-4 --itr 5 

############################predict length 96####################################
python main_crossformer.py --data ETTm1 \
--in_len 672 --out_len 96 --seg_len 12 \
--learning_rate 1e-4 --itr 5 

############################predict length 288####################################
python main_crossformer.py --data ETTm1 \
--in_len 672 --out_len 288 --seg_len 24 \
--learning_rate 1e-5 --itr 5 

############################predict length 672####################################
python main_crossformer.py --data ETTm1 \
--in_len 672 --out_len 672 --seg_len 12 \
--learning_rate 1e-5 --itr 5 