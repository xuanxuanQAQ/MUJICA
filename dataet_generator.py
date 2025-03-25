import dataset

task = 'channel_estimation'

if task == 'channel_estimation':
    input_type = "pilot"
    label_type = "mpsk_signal"
else:
    input_type = "normalized_signal"
    label_type = "bits"

dataset.generate_train_data(batch_size=2, num_batch=4, root_dir="data/", dataset_type="train", num_bits=2048, poly='CRC-16', m_psk=4, n_fft=64, comb_num=8,
                    n_cp=16, sample_rate=150, snr_db=50, wind_speed=1.5, label=label_type, input=input_type, use_hdf5=False, use_multiprocessing=False)
