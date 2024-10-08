import numpy as np
from scipy.signal import butter, filtfilt, welch, firwin, lfilter, spectrogram
import matplotlib.pyplot as plt

def chrom_rppg(rgb_signal, fs, cutoff_freq=[0.5, 2.5]):
    """
    Estimate the remote photoplethysmography (rPPG) signal using the CHROM algorithm with bandpass filtering.
    
    Parameters:
    rgb_signal (numpy.ndarray): A numpy array of shape (3, temporal) representing the RGB signals.
    fs (float): The sampling frequency of the signal in Hz.
    cutoff_freq (list): A list with two elements representing the low and high cutoff frequencies for the bandpass filter.
    
    Returns:
    numpy.ndarray: A 1D numpy array representing the filtered rPPG signal.
    """
    # Ensure the input has the correct shape
    assert rgb_signal.shape[0] == 3, "Input must have 3 channels (RGB)"
    
    # Normalize the RGB signals (remove the mean)
    rgb_signal = rgb_signal - np.mean(rgb_signal, axis=1, keepdims=True)
    
    # Compute the CHROM signals X and Y
    X = 3 * rgb_signal[0] - 2 * rgb_signal[1]
    Y = 1.5 * rgb_signal[0] + rgb_signal[1] - 1.5 * rgb_signal[2]
    
    # Compute the rPPG signal S as the linear combination of X and Y
    alpha = np.std(X) / np.std(Y)
    S = X - alpha * Y
    
    # Normalize the output signal
    S = S / np.std(S)
    
    # Design a Butterworth bandpass filter
    nyquist_freq = 0.5 * fs
    low_cutoff = cutoff_freq[0] / nyquist_freq
    high_cutoff = cutoff_freq[1] / nyquist_freq
    b, a = butter(N=4, Wn=[low_cutoff, high_cutoff], btype='bandpass')
    
    # Apply the filter to the signal
    S_filtered = filtfilt(b, a, S)
    
    return S_filtered

def select_most_informative_signal(rppg_signals, fs=30, visualize=False):
    '''
    Select the most informative rPPG signal based on Welch's method for PSD computation.
    
    Input:
        - rppg_signals: numpy.ndarray, shape (3, temporal), three rPPG signals to choose from.
        - fs: int, the sampling frequency (default: 30Hz).
        - visualize: bool, if True, plots the PSD and spectrogram of the signals.
    
    Output:
        - best_signal_index: int, the index of the most informative signal (0, 1, or 2).
    '''
    
    # Define the physiological frequency range of interest (0.5 to 4 Hz)
    freq_range = (0.5, 4.0)
    
    # Initialize variables to track the best signal
    best_signal_index = None
    max_peak_power = -np.inf
    peak_powers = []  # To store peak powers for visualization
    
    # Iterate over each signal to compute its PSD and find the most informative one
    for i in range(rppg_signals.shape[0]):
        # Compute the Power Spectral Density (PSD) using Welch's method
        freqs, psd = welch(rppg_signals[i], fs=fs, nperseg=fs*4)
        
        # Limit the PSD to the physiological frequency range
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        psd_in_range = psd[mask]
        
        # Find the peak power in the physiological range
        peak_power = np.max(psd_in_range)
        peak_powers.append(peak_power)
        
        # Check if this signal has the highest peak power
        if peak_power > max_peak_power:
            max_peak_power = peak_power
            best_signal_index = i
    
    # Visualization of the PSD and spectrogram
    if visualize:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        for i in range(rppg_signals.shape[0]):
            # Plot the PSD
            freqs, psd = welch(rppg_signals[i], fs=fs, nperseg=fs*4)
            axes[0, i].plot(freqs, psd, label=f'Signal {i + 1}')
            axes[0, i].axvline(x=freqs[np.argmax(psd)], color='r', linestyle='--', alpha=0.6)
            axes[0, i].text(freqs[np.argmax(psd)], np.max(psd), f'Peak: {peak_powers[i]:.2f}', 
                            verticalalignment='bottom', horizontalalignment='left')
            axes[0, i].set_xlim(freq_range)
            axes[0, i].set_title(f'PSD of Signal {i + 1}')
            axes[0, i].set_xlabel('Frequency (Hz)')
            axes[0, i].set_ylabel('Power Spectral Density')
            axes[0, i].grid(True)
            axes[0, i].legend()
            
            # Plot the Spectrogram
            f, t, Sxx = spectrogram(rppg_signals[i], fs=fs, nperseg=fs*4, noverlap=fs*2, window='hamming')
            im = axes[1, i].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
            axes[1, i].set_ylim(freq_range)
            axes[1, i].set_title(f'Spectrogram of Signal {i + 1}')
            axes[1, i].set_xlabel('Time (s)')
            axes[1, i].set_ylabel('Frequency (Hz)')
            fig.colorbar(im, ax=axes[1, i], format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig('./out/compare_signal.png')
        plt.show()
        
    
    return best_signal_index



def bandpass_filter(data, fs=30, lowcut=0.6, highcut=2.0, numtaps=101):
    '''
    Apply a bandpass filter to the mean pixel value data.
    
    Input:
        - data: numpy.ndarray, shape (3, 3, temporal), the mean pixel values across ROIs and channels.
        - fs: int, sampling frequency in Hz (default: 30Hz).
        - lowcut: float, lower cutoff frequency in Hz (default: 0.6Hz).
        - highcut: float, upper cutoff frequency in Hz (default: 2.0Hz).
        - numtaps: int, number of taps (coefficients) in the FIR filter (default: 101).
    
    Output:
        - filtered_data: numpy.ndarray, shape (3, 3, temporal), the filtered mean pixel values.
    '''
    
    # Design the bandpass filter using a Hamming window
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    taps = firwin(numtaps, [low, high], pass_zero=False, window='hamming')
    
    # Initialize the output array
    filtered_data = np.zeros_like(data)
    
    # Apply the filter to each ROI and channel
    for i in range(data.shape[0]):  # Loop over ROIs
        for j in range(data.shape[1]):  # Loop over channels
            # Apply the filter to the temporal data
            filtered_data[i, j, :] = lfilter(taps, 1.0, data[i, j, :])
    
    return filtered_data

