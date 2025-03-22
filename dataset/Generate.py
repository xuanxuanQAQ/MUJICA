import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import simulation
import modulation

def datasetGenerate(num_bits = 2048, poly = 'CRC-16', m_psk = 4, n_fft = 64, n_cp = 16, sample_rate = 150, shif_freq = 300, ):
    
    
    bits = simulation.generate_random_binary(num_bits)
    crc_bits = modulation.add_crc(bits, poly)
    complex_symbols = modulation.mpsk_modulation(crc_bits, m_psk)
    
    ofdm_signal, frame_structure = modulation.ofdm_modulation(complex_symbols, n_fft, n_cp)
    real_ofdm_signal = np.real(ofdm_signal)
    
    time_indices = modulation.generate_time_indices(real_ofdm_signal, sample_rate, 0)
    signal_shifted = modulation.frequency_shift_real(real_ofdm_signal, time_indices, shif_freq, None)
    
    
    
    
    
    return bits, complex_symbols
    
    