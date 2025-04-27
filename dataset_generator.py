import dataset
import deeplearning as dl

# 参数
num_batch = 120
root_dir = "data/"
dataset_type = "train"
num_bits = 2048
poly = 'CRC-16'
m_psk = 4
n_fft = 64
comb_num = 8
n_cp = 16
sample_rate = 150
snr_db = 50
wind_speed = 1.5
use_hdf5 = False
use_multiprocessing = False

task = 'channel_equalizer'

methods = 'dl'

if task == 'channel_estimation':
    input_type = "ofdm_freq_symbols"
    label_type = "mpsk_signal"
elif task == 'channel_equalizer':
    input_type = "ofdm_freq_symbols"
    label_type = "mpsk_signal"
elif task == 'filter':
    input_type = "recieved_signal"
    label_type = "ofdm_signal"

dataset.generate_train_data(num_batch, root_dir, dataset_type, num_bits, poly, m_psk, n_fft, comb_num, 
                             n_cp, sample_rate, snr_db, wind_speed, label_type, input_type, use_hdf5, use_multiprocessing)

if task =='channel_equalizer':
    input_type = "ofdm_freq_symbols"
    output_type = "channel_estimates"
    
    model = dl.Transformer(
        input_dim=2,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3
    )
    
    dataset.generate_joint_data(num_batch, root_dir, dataset_type, input_type, output_type, methods, model, use_hdf5, use_multiprocessing)

