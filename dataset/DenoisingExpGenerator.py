import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import glob
import radar
import demodulation
import simulation
import utils


def denoising_bpsk_exp_generator(folder = 'data/exp', output_input_dir = 'data/bpsk_exp_input', output_label_dir = 'data/bpsk_exp_label'):
    
    # 确保输出目录存在
    os.makedirs(output_input_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    fc = 200  # str2double(fileName(6:8))
    FrameNum = list(range(1, 257))  # 1:256 in MATLAB
    lamda = 3e8 / 77e9
    ChannlNum = 0  # Python is 0-indexed, MATLAB is 1-indexed
    Rb = 100
    fc = 200  # Rb: 码元速率
    modulationIndex = fc / Rb  # Modulate at one bit per two cycles

    # 指定包含.bin文件的文件夹路径
    file_pattern = os.path.join(folder, '*BP*SK*.bin')
    files = glob.glob(file_pattern)  # 获取文件夹中所有匹配的.bin文件的列表

    for file_path in files:
        file_name = os.path.basename(file_path)
        file_base_name = os.path.splitext(file_name)[0]
        
        # 检查输出文件是否已存在
        input_csv_path = os.path.join(output_input_dir, f"{file_base_name}_input.csv")
        label_csv_path = os.path.join(output_label_dir, f"{file_base_name}_label.csv")
        
        if os.path.exists(input_csv_path) and os.path.exists(label_csv_path):
            print(f"跳过已处理的文件: {file_path}")
            continue
        
        rawData = radar.read_dca1000(file_path)
        
        params = radar.radar_params_extract(file_path)
        ADCSample, ChirpPeriod, ADCFs, ChirpNum, FramPeriod, FramNum, slope, BandWidth, R_Maximum, R_resulo, V_Maximum, V_resulo = params
        fs = 1e6 / ChirpPeriod

        Len = rawData.shape[1]
        fullChirp = FramPeriod / ChirpPeriod
        
        times, times_compen = radar.create_time_arrays(ChirpPeriod, FrameNum, fullChirp)
        
        ChannlNum = 0  
        
        frames_dimension = int(round(Len/(ADCSample*ChirpNum)))
        Data_all = np.reshape(rawData, (4, int(ADCSample), int(ChirpNum), frames_dimension), order='F')
        proData = np.reshape(Data_all[:, :, :, np.array(FrameNum)-1], (4, int(ADCSample), -1), order='F')
        
        DataRangeFft, _ = radar.range_fft(proData, int(ADCSample), BandWidth, apply_window=False)
        
        processed_phases = []
        rxDatas = []
        ralocs = []
        for ChannlNum in range(4):    
            _, maxlocAll = radar.find_max_energy_range_bin(DataRangeFft[ChannlNum, :, :])
            phase_range = radar.extract_phase_from_max_range_bin(DataRangeFft, maxlocAll, range_search=3, channel_num=ChannlNum, time_increment=1)
            
            # Process max power range bin 
            unwrapped_phase, _ = radar.extract_and_unwrap_phase(phase_range)
            processed_phase, _, _ = radar.process_micro_phase(unwrapped_phase, times, times_compen, window_size=57, poly_order=3, threshold=0.02)
            
            # BPSK demodulation using two methods
            _, rxData, _, _, raloc = demodulation.bpsk_demodulator_with_symbol_sync(fs, fc, modulationIndex, processed_phase)
            
            processed_phases.append(processed_phase)
            rxDatas.append(rxData)
            ralocs.append(raloc)
            
        processed_phases = np.array(processed_phases)    
        
        
        decoded_signal = utils.majority_vote_decoder(rxDatas)
        # Calculate error rates
        aligned_pattern, error = demodulation.Error110Func(decoded_signal)
        if error > 0.4:
            print(f"文件 {file_path} 的误比特率过高: {error}")
            continue
        
        print(f"File: {file_path}, Error Rate: {error}")
        
        label, _ = simulation.generate_bpsk_signal(fs, fc, modulationIndex, bit_pattern=aligned_pattern)
        input_data = utils.trim_input_with_label(processed_phases, label, raloc)
        
        # 保存为CSV文件
        # 保存input数据 - 每个通道作为一行
        input_csv_path = os.path.join(output_input_dir, f"{file_base_name}_input.csv")
        np.savetxt(input_csv_path, input_data, delimiter=',')
        
        # 保存label数据 - 单行数据
        label_csv_path = os.path.join(output_label_dir, f"{file_base_name}_label.csv")
        np.savetxt(label_csv_path, label.reshape(1, -1), delimiter=',')
        
        print(f"已保存 {input_csv_path} 和 {label_csv_path}")