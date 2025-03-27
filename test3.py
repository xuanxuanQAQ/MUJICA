import deeplearning as dl
# import multiprocessing
# multiprocessing.freeze_support()

model = dl.Transformer(
        input_dim=2,  # 复数信号 (实部+虚部)
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3
    )

# Load the data
data = dl.load_channel_data(
        signal_dir='data/Train_mpsk_signal_B00', 
        pilot_dir='data/Train_pilot_B00',
        structure_dir='data/train_frame_structure.json',
        batch_size=4
    )

dl.train_model(model, data, num_epochs=10, learning_rate=0.001)