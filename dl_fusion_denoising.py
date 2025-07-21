import deeplearning as dl
import radar
import demodulation
import utils

# import multiprocessing
# multiprocessing.freeze_support()

config = 'predict' # 'train' or 'predict'
method = 'conv' # 'conv' or 'GRU'

save_dir = "model/fusion_denoising"
channel_est_pt = 'model/fusion_denoising/best_model.pth'

if method == 'conv':
    model = dl.TimeFrequencyFilterNet(
        hidden_dim=64, 
        n_fft=256, 
        hop_length=64
    )

if config == "train":
    
    if method == 'conv':
        # Load the data
        data = dl.load_fusion_denoising_exp_data(
                input_dir='data/bpsk_exp_input',
                label_dir='data/bpsk_exp_label',
                batch_size=4
            )

        dl.train_fusion_denoising(model, data, epochs=300, learning_rate=0.002, save_dir=save_dir)
        
elif config =="predict":
    
    file_name = 'BPSKRb100A110Fc200P70F1_Sbig2_Raw_0.bin'
    processed_phase, fs, fc, modulationIndex = radar.extract_processed_radar_phase(file_name)
    
    rxDatas = []
    for i in range(4):
        _, rxData, _, _, raloc = demodulation.bpsk_demodulator_with_symbol_sync(fs, fc, modulationIndex, processed_phase[i])
        rxDatas.append(rxData)
    
    decoded_signal = utils.majority_vote_decoder(rxDatas)
    _, error = demodulation.Error110Func(decoded_signal)
    print(f"File: {file_name}, Standard Error Rate: {error}")
    
    if method == 'conv':
        
        denoised_phase = dl.predict_fusion_denoising(model, processed_phase, 'model/fusion_denoising/best_model.pth')
        
        _, rxData_dl, _, _, raloc = demodulation.bpsk_demodulator_with_symbol_sync(fs, fc, modulationIndex, denoised_phase)
        _, error_dl = demodulation.Error110Func(rxData_dl)
        print(f"File: {file_name}, DL Error Rate: {error_dl}")
        


