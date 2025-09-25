############################predict length 24####################################
python main_crossformer.py --data ILI \
--in_len 48 --out_len 24 --seg_len 6 \
--e_layers 2 \
--learning_rate 5e-4 --dropout 0.6 --itr 5 

############################predict length 36####################################
python main_crossformer.py --data ILI \
--in_len 48 --out_len 36 --seg_len 6 \
--e_layers 2 \
--learning_rate 5e-4 --dropout 0.6 --itr 5 

############################predict length 48####################################
python main_crossformer.py --data ILI \
--in_len 60 --out_len 48 --seg_len 6 \
--e_layers 2 \
--learning_rate 5e-4 --dropout 0.6 --itr 5  

############################predict length 60####################################
python main_crossformer.py --data ILI \
--in_len 60 --out_len 60 --seg_len 6 \
--e_layers 2 \
 --learning_rate 5e-4 --dropout 0.6 --itr 5 