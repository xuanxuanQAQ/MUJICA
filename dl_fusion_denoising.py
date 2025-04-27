import deeplearning as dl

# import multiprocessing
# multiprocessing.freeze_support()

config = 'train'
method = 'GRU' # 'conv' or 'transformer' or 'GRU'

save_dir = "model/fusion_denoising"
channel_est_pt = 'model/fusion_denoising/best_model.pth'

if method == 'conv':
    model = dl.TimeFrequencyFilterNet(
        hidden_dim=64, 
        n_fft=256, 
        hop_length=64
    )
elif method == 'transformer':
    model = dl.TimeFrequencyTransformerNet(
        hidden_dim=64,
        n_fft=256,
        hop_length=64,
        in_channels=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        nhead=4,
        dropout=0.1
    )
elif method == 'GRU':
    model = dl.TimeFrequencyBiGRUNet(
        hidden_dim=32, 
        n_fft=256, 
        hop_length=64, 
        in_channels=4,
        num_gru_layers=1, 
        dropout=0.1,
        window_size=512, 
        window_stride=256
    )

if config == "train":
    
    if method == 'conv':
        # Load the data
        data = dl.load_fusion_denoising_exp_data(
                input_dir='data/bpsk_exp_input',
                label_dir='data/bpsk_exp_label',
                batch_size=4
            )

        dl.train_fusion_denoising(model, data, epochs=600, learning_rate=0.002, save_dir=save_dir)
        
    elif method == 'transformer':      
        # Load the data
        data = dl.load_fusion_denoising_exp_data(
                input_dir='data/bpsk_exp_input',
                label_dir='data/bpsk_exp_label',
                batch_size=1
            )

        dl.train_fusion_denoising_transformer(model, data, epochs=1000, learning_rate=0.002, save_dir=save_dir)
        
    elif method == 'GRU':
        # Load the data
        data = dl.load_fusion_denoising_exp_data(
                input_dir='data/bpsk_exp_input',
                label_dir='data/bpsk_exp_label',
                batch_size=4
            )

        dl.train_fusion_denoising_bigru(model, data, epochs=1000, learning_rate=0.002, save_dir=save_dir)
    
elif config =="predict":
    
    input()