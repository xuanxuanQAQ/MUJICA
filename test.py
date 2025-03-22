import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def sliding_window(signal, window_size, overlap_percent=50):
    """
    使用滑动窗口技术将复数信号分割成重叠窗口。
    
    参数:
        signal (np.ndarray): 输入信号，可以是以下形式:
                            - 复数数组 (n_samples,) 或 (n_samples, n_features)
                            - 实部和虚部分离的数组 (n_samples, 2) 或 (n_samples, n_features, 2)
        window_size (int): 每个窗口的样本数
        overlap_percent (float): 重叠百分比，默认50%
    
    返回:
        np.ndarray: 窗口化后的信号，形状为:
                    - 对于复数输入: (n_windows, window_size, n_features) 复数数组
                    - 对于分离实虚部输入: (n_windows, window_size, n_features, 2)
    """
    is_complex = is_complex_before = np.iscomplexobj(signal)
    
    if is_complex:
        if len(signal.shape) == 1:
            signal = signal.reshape(-1, 1)
    else:
        if len(signal.shape) == 2 and signal.shape[1] == 2:
            signal = signal[:, 0] + 1j * signal[:, 1]
            signal = signal.reshape(-1, 1)
            is_complex = True
        elif len(signal.shape) == 3 and signal.shape[2] == 2:
            signal = signal[:, :, 0] + 1j * signal[:, :, 1]
            is_complex = True
    
    n_samples, n_features = signal.shape
    
    step = int(window_size * (1 - overlap_percent/100))
    step = max(1, step) 

    n_windows = (n_samples - window_size) // step + 1
    
    if is_complex:
        windows = np.zeros((n_windows, window_size, n_features), dtype=complex)
    else:
        windows = np.zeros((n_windows, window_size, n_features))
    
    for i in range(n_windows):
        start_idx = i * step
        end_idx = start_idx + window_size
        windows[i] = signal[start_idx:end_idx]
    
    if is_complex and not is_complex_before:
        real_part = np.real(windows)
        imag_part = np.imag(windows)
        windows_separate = np.concatenate((real_part, imag_part), axis=-1)
        return windows_separate
    
    if n_features == 1 and len(signal.shape) == 2:
        return windows.reshape(n_windows, window_size)
    
    return windows

def complex_to_channels(complex_signal):
    """
    将复数信号转换为双通道表示，分离实部和虚部。
    
    参数:
        complex_signal (np.ndarray): 输入的复数信号，可以是以下形状:
                                    - 一维数组 (n_samples,)
                                    - 二维数组 (n_samples, n_features)
                                    
    返回:
        np.ndarray: 转换后的双通道实数数组，形状为:
                    - 对于一维输入: (n_samples, 2)
                    - 对于二维输入: (n_samples, n_features, 2)
                    其中最后一维度的索引0是实部，索引1是虚部
    """
    if not np.iscomplexobj(complex_signal):
        raise ValueError("输入必须是复数数组")
    
    if len(complex_signal.shape) == 1:
        real_part = np.real(complex_signal)
        imag_part = np.imag(complex_signal)
        return np.stack((real_part, imag_part), axis=1)
    
    elif len(complex_signal.shape) == 2:
        real_part = np.real(complex_signal)
        imag_part = np.imag(complex_signal)
        return np.stack((real_part, imag_part), axis=2)
    
    else:
        raise ValueError("仅支持一维或二维复数数组")

def channels_to_complex(channel_signal):
    """
    将双通道表示（实部和虚部）转回复数信号。
    
    参数:
        channel_signal (np.ndarray): 输入的双通道实数信号，可以是以下形状:
                                    - (n_samples, 2) 其中[:,0]是实部，[:,1]是虚部
                                    - (n_samples, n_features, 2)
    
    返回:
        np.ndarray: 转换后的复数数组，形状为:
                    - 对于 (n_samples, 2) 输入: (n_samples,)
                    - 对于 (n_samples, n_features, 2) 输入: (n_samples, n_features)
    """
    if not (len(channel_signal.shape) >= 2 and channel_signal.shape[-1] == 2):
        raise ValueError("输入的最后一个维度必须是2（表示实部和虚部）")
    
    if len(channel_signal.shape) == 2:
        return channel_signal[:, 0] + 1j * channel_signal[:, 1]
    
    elif len(channel_signal.shape) == 3:
        return channel_signal[:, :, 0] + 1j * channel_signal[:, :, 1]
    
    else:
        raise ValueError("仅支持(n_samples, 2)或(n_samples, n_features, 2)形状的输入")

def generate_complex_sine_wave(fs=1000, duration=1.0, 
                              freq_range=(5, 50), 
                              amplitude_range=(0.5, 2.0),
                              phase_range=(0, 2*np.pi),
                              noise_level=0.1,
                              num_components=3):
    """
    生成包含多个随机频率分量的复数正弦波信号
    
    参数:
        fs (float): 采样率 (Hz)
        duration (float): 信号持续时间 (秒)
        freq_range (tuple): 频率范围 (Hz) 的最小值和最大值
        amplitude_range (tuple): 振幅范围的最小值和最大值
        phase_range (tuple): 相位范围 (弧度) 的最小值和最大值
        noise_level (float): 添加到信号的噪声水平
        num_components (int): 要添加到信号中的正弦波分量数量
        
    返回:
        tuple: (complex_signal, time_axis)
            - complex_signal: 复数信号数组
            - time_axis: 时间轴
    """
    # 创建时间轴
    t = np.arange(0, duration, 1/fs)
    num_samples = len(t)
    
    # 初始化复数信号
    complex_signal = np.zeros(num_samples, dtype=complex)
    
    # 生成多个随机正弦波分量并叠加
    for _ in range(num_components):
        # 随机参数
        freq = np.random.uniform(*freq_range)
        amplitude = np.random.uniform(*amplitude_range)
        phase = np.random.uniform(*phase_range)
        
        # 生成复数正弦波 (Ae^(j(2πft + φ)))
        component = amplitude * np.exp(1j * (2 * np.pi * freq * t + phase))
        complex_signal += component
    
    # 添加复数高斯噪声
    if noise_level > 0:
        noise_real = np.random.normal(0, noise_level, num_samples)
        noise_imag = np.random.normal(0, noise_level, num_samples)
        complex_noise = noise_real + 1j * noise_imag
        complex_signal += complex_noise
    
    return complex_signal, t

class testNeuralNetwork(nn.Module):
    def __init__(self, feature_dim):
        super(testNeuralNetwork, self).__init__()
        
        self.batch_norm1 = nn.BatchNorm1d(num_features=2)
        
        self.conv1d1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding='same')
        self.conv1d2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same')
        
        self.mish = nn.Mish()
        
        self.bigru1 = nn.GRU(input_size=16, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.bigru2 = nn.GRU(input_size=128, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        
        self.dim_reduction1 = nn.Conv1d(in_channels=512, out_channels=16, kernel_size=1)
        
        self.bigru3 = nn.GRU(input_size=16, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        
        self.dim_reduction2 = nn.Conv1d(in_channels=128, out_channels=16, kernel_size=1)
        
        self.conv1d3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same')
        
        self.layer_norm1 = nn.LayerNorm(32)
        
        self.dim_reduction3 = nn.Conv1d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding='same')
        
        
        
        
    def forward(self, x):
        
        x = x.transpose(1, 2)   # [batch_size, channels, length]
        x = self.batch_norm1(x)
        
        x_res = x = self.conv1d1(x)
        
        x = self.conv1d2(x)
        x = self.mish(x)
        
        x = x.transpose(1, 2)   # [batch_size, length, channels]
        x, _  = self.bigru1(x)
        x, _  = self.bigru2(x)
        x = x.transpose(1, 2)   # [batch_size, channels, length]
        x = self.dim_reduction1(x)
        
        x_res = x = x_res + x
        
        x = x.transpose(1, 2)   # [batch_size, length, channels]
        x, _  = self.bigru3(x)
        x = self.dim_reduction2(x)
        x = self.mish(x)
        
        x += x_res
        
        x = self.conv1d3(x)
        x = self.conv1d3(x)
        
        return x
        
def train(model, train_loader, val_loader=None, epochs=100, lr=0.001, device='cuda', 
          early_stopping_patience=10, scheduler_patience=5):
    """
    Training function for the neural network.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to run training on ('cuda' or 'cpu')
        early_stopping_patience: Number of epochs to wait before early stopping
        scheduler_patience: Number of epochs to wait before reducing learning rate
    
    Returns:
        model: Trained model
        history: Dictionary containing training and validation losses
    """
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience, verbose=True
    )
    
    # Initialize history dictionary to store losses
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, targets in progress_bar:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # For visualization during training - ensure correct dimensions
            outputs = outputs.transpose(1, 2)  # [batch_size, length, channels]
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({'train_loss': loss.item()})
        
        # Calculate average training loss for the epoch
        train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation phase (if validation data is provided)
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            
            # Print epoch statistics
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}')
            
            # Learning rate scheduler step
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                # Load best model
                model.load_state_dict(best_model_state)
                break
        else:
            # Print epoch statistics without validation
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}')
            scheduler.step(train_loss)
    
    # If validation was used, ensure we return the best model
    if val_loader is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model on the given data loader.
    
    Args:
        model: The neural network model
        data_loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to run evaluation on ('cuda' or 'cpu')
    
    Returns:
        float: Average loss
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # For visualization during evaluation - ensure correct dimensions
            outputs = outputs.transpose(1, 2)  # [batch_size, length, channels]
            
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
    
    return total_loss / len(data_loader.dataset)

def predict(model, inputs, device='cuda'):
    """
    Make predictions using the trained model.
    
    Args:
        model: The trained neural network model
        inputs: Input data (torch.Tensor)
        device: Device to run prediction on ('cuda' or 'cpu')
    
    Returns:
        torch.Tensor: Model predictions
    """
    model.eval()
    model = model.to(device)
    
    # Move input to device
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs).float()
    
    if len(inputs.shape) == 2:  # Single sample
        inputs = inputs.unsqueeze(0)  # Add batch dimension
    
    inputs = inputs.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(inputs)
        outputs = outputs.transpose(1, 2)  # [batch_size, length, channels]
    
    return outputs.cpu().numpy()

def prepare_data(X, y, batch_size=32, val_split=0.2, shuffle=True):
    """
    Prepare data loaders for training and validation.
    
    Args:
        X: Input features (numpy array or torch.Tensor)
        y: Target values (numpy array or torch.Tensor)
        batch_size: Batch size for DataLoader
        val_split: Fraction of data to use for validation
        shuffle: Whether to shuffle the data
    
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
    """
    # Convert numpy arrays to torch tensors if needed
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()
    
    # Determine split index
    val_size = int(len(X) * val_split)
    train_size = len(X) - val_size
    
    # Split the data
    if shuffle:
        indices = torch.randperm(len(X))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
    else:
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:], y[train_size:]
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def plot_training_history(history):
    """
    Plot training and validation loss history.
    
    Args:
        history: Dictionary containing training and validation losses
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate sample data (replace with your actual data)
    # Example: X shape should be [samples, features, timesteps] or [samples, timesteps, features]
    # Example: y shape should match your model's output
    feature_dim = 2
    sequence_length = 100
    num_samples = 1000
    
    X = np.random.randn(num_samples, sequence_length, feature_dim).astype(np.float32)
    y = np.random.randn(num_samples, sequence_length, feature_dim).astype(np.float32)
    
    # Prepare data loaders
    train_loader, val_loader = prepare_data(X, y, batch_size=32)
    
    # Initialize model
    model = testNeuralNetwork(feature_dim=feature_dim)
    
    # Train model
    trained_model, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        lr=0.001,
        device=device
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Make predictions
    sample_input = X[:5]  # Take first 5 samples
    predictions = predict(trained_model, sample_input, device)
    
    print(f"Predictions shape: {predictions.shape}")
    
    # Save model
    torch.save(trained_model.state_dict(), 'trained_model.pth')
    print("Model saved successfully!")
    



# 使用示例
if __name__ == "__main__":
    window_size = 200
    
    signal, t  = generate_complex_sine_wave()
    signal_channel = complex_to_channels(signal)
    windowed_signal = sliding_window(signal_channel, window_size, 50)
    print(f"创建的窗口数量: {len(windowed_signal[0])}")
    print(f"每个窗口的形状: {windowed_signal[0].shape}")
    
    
    
    
    