
python main.py --experiment_description _reg \
--run_description SHHS_100_To_EDF20 \
--encoder MMASleepNet_EEG --seed 123 \
--source_path /disk2/data_npy/shhs_reg_all/ \
--target_path /disk2/data_npy/sleepedf-78-reg/ \
--train_mode TUDAMatch \
--logs_save_dir experiment \
--discriminator Discriminator

