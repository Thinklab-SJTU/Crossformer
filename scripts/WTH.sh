############################predict length 24####################################
python main_crossformer.py --data WTH \
--in_len 48 --out_len 24 --seg_len 6 \
--learning_rate 1e-4 --itr 5 

############################predict length 24####################################
python main_crossformer.py --data WTH \
--in_len 48 --out_len 48 --seg_len 12 \
--learning_rate 1e-4 --itr 5 

############################predict length 168###################################
python main_crossformer.py --data WTH \
--in_len 336 --out_len 168 --seg_len 24 \
--learning_rate 1e-5 --itr 5 

############################predict length 336###################################
python main_crossformer.py --data WTH \
--in_len 336 --out_len 336 --seg_len 24 \
--learning_rate 1e-5 --itr 5 

############################predict length 720###################################
python main_crossformer.py --data WTH \
--in_len 720 --out_len 720 --seg_len 24 \
--learning_rate 1e-5 --itr 5 