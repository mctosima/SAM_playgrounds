import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
import json
import scipy.signal as signal
import heartpy as hp
from scipy.signal import firwin, lfilter
import pandas as pd

DATASET_PATH = os.path.join('masked_PURE')



def calculate_ac_dc(data):
    """
    Calculate AC (standard deviation) and DC (mean) for RED and BLUE channels.
    
    Parameters:
    data (numpy array): A numpy array with shape (3, 1961)
    
    Returns:
    tuple: A tuple containing (ACRed, ACBlue, DCRed, DCBlue)
    """
    # Assuming data[0] is RED, data[2] is BLUE, and data[1] is GREEN
    red_channel = data[1]
    blue_channel = data[2]
    
    # Calculate AC (standard deviation) and DC (mean) for RED channel
    ACRed = np.std(red_channel)
    DCRed = np.mean(red_channel)
    
    # Calculate AC (standard deviation) and DC (mean) for BLUE channel
    ACBlue = np.std(blue_channel)
    DCBlue = np.mean(blue_channel)
    
    return ACRed, ACBlue, DCRed, DCBlue

def calculate_spo2(data, A, B):
    """
    Calculate SpO2 using the given formula.
    
    Parameters:
    data (numpy array): A numpy array with shape (3, 1961)
    A (float): Calibration parameter A
    B (float): Calibration parameter B
    
    Returns:
    float: Calculated SpO2 value
    """
    # Calculate AC and DC values for RED and BLUE channels
    ACRed, ACBlue, DCRed, DCBlue = calculate_ac_dc(data)
    
    # Calculate SpO2 using the given formula
    ror_red = ACRed / DCRed
    ror_blue = ACBlue / DCBlue
    spo2 = A - B * (ror_red / ror_blue)
    
    print(f"ror_red: {ror_red} | ror_blue: {ror_blue} | ror_red/ror_blue: {ror_red/ror_blue}")
    
    return spo2

def get_gt(subject_name: str):
    json_path = os.path.join(DATASET_PATH, "gt", f'{subject_name}.json')
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    gt_val = [frame["Value"]["o2saturation"] for frame in json_data["/FullPackage"]]
    return gt_val

def bandpass_filter(data, cutoff_freqs, order=3, fs=30):
    
    b, a = signal.butter(order, [cutoff_freqs[0], cutoff_freqs[1]], btype='bandpass', fs=fs)
    data_filtered = []
    
    for each_signal in data:
        filtered_signal = signal.filtfilt(b, a, each_signal)
        data_filtered.append(filtered_signal)
        
    return np.array(data_filtered)

def highpass_filter_multichannel(data, sample_rate, cutoff_freq, order=4):
    """
    Apply a high-pass filter to multi-channel data while preserving amplitude and offset.
    
    Parameters:
    data (numpy.ndarray): Input signal with shape (channels, time_signal)
    sample_rate (float): Sampling rate of the signal in Hz
    cutoff_freq (float): Cutoff frequency for the high-pass filter in Hz
    order (int): Order of the Butterworth filter (default is 4)
    
    Returns:
    numpy.ndarray: Filtered signal with preserved amplitude and offset
    """
    num_channels, signal_length = data.shape
    
    # Normalize the cutoff frequency
    nyquist_freq = 0.5 * sample_rate
    normalized_cutoff = cutoff_freq / nyquist_freq
    
    # Design the Butterworth high-pass filter
    b, a = signal.butter(order, normalized_cutoff, btype='high', analog=False)
    
    # Initialize the output array
    filtered_data = np.zeros_like(data)
    
    # Apply the filter to each channel
    for i in range(num_channels):
        # Store original min, max, and mean
        original_min = np.min(data[i, :])
        original_max = np.max(data[i, :])
        original_mean = np.mean(data[i, :])
        
        # Apply the filter
        filtered_channel = signal.filtfilt(b, a, data[i, :])
        
        # Adjust the filtered signal to match original range
        filtered_min = np.min(filtered_channel)
        filtered_max = np.max(filtered_channel)
        
        # Scale and offset to match original range
        scaled_filtered = (filtered_channel - filtered_min) / (filtered_max - filtered_min)
        scaled_filtered = scaled_filtered * (original_max - original_min) + original_min
        
        # Restore the original offset
        filtered_data[i, :] = scaled_filtered + (original_mean - np.mean(scaled_filtered))
    
    return filtered_data

def sliding_average(data, window_size=10):
    """
    Calculate the sliding average of the given data.
    
    Parameters:
    data (numpy array): A numpy array with shape (3, 1961)
    window_size (int): The size of the window for the sliding average
    
    Returns:
    numpy array: A numpy array containing the sliding average of the given data
    """
    # Calculate the sliding average
    data_avg = np.zeros_like(data)
    for i in range(data.shape[1]):
        start = max(0, i - window_size)
        end = min(data.shape[1], i + window_size)
        data_avg[:, i] = np.mean(data[:, start:end], axis=1)
    
    return data_avg
    
def fir_filter(signal, order=101, cutoff=0.1):
    fir_coeff = firwin(order, cutoff)
    for i in range(3):
        signal[i] = lfilter(fir_coeff, 1.0, signal[i])
    return signal

def main(
    start_from_subject: int = 0, 
    end_at_subject: int = None,
    plot_mean_rgb: bool = False,
    cutoff_freqs: list = None,
    sliding_window_size: int = None,
    scenario: int = None,
    ):
    # List all the files in the dataset
    npy_files = glob(os.path.join(DATASET_PATH, '*.npy'))
    file_names = [os.path.splitext(os.path.basename(file))[0] for file in npy_files]
    file_names.sort()
    print(f"Total files: {len(file_names)} | Files: {file_names}")
    
    # Filter specific scenario
    if scenario is not None:
        file_names = [file for file in file_names if file.endswith(f"-0{scenario}")]
    
    # Start from a specific subject
    if end_at_subject is not None:
        file_names = file_names[start_from_subject:end_at_subject]
    else:
        file_names = file_names[start_from_subject:]
    
    # Loop through all the files
    err_list = []
    pred_list = []
    gt_list = []
    for i,subject in enumerate(file_names):
        print(f"Processing {subject} | ({i+1}/{len(file_names)})")
        # Load the numpy file
        path = os.path.join(DATASET_PATH, f'{subject}.npy')
        data = np.load(path)
        
        # Get GT
        gt_val = get_gt(subject)
        
        # Switch between the first and third channels
        data = data[...,[2,1,0]] # from BGR to RGB
        
        # Try to plot the first frame
        if False:
            plt.figure(figsize=(12, 6))
            plt.imshow(data[0])
            plt.axis('off')
            plt.title(f'First frame of {subject}')
            plt.tight_layout()
            plt.savefig(os.path.join('out', f'sample_frame.png'))
        
        # Extract the mean RGB values
        mean_rgb = np.mean(data, axis=(1, 2))
        mean_rgb = mean_rgb.T
        
        # fir filter
        # mean_rgb = fir_filter(mean_rgb, order=101, cutoff=0.1)
        
        # normalize signal to 0-1
        # for i in range(3):
        #     mean_rgb[i] = (mean_rgb[i] - np.min(mean_rgb[i])) / (np.max(mean_rgb[i]) - np.min(mean_rgb[i]))
        
        # Try to do sliding average on the mean RGB values
        if sliding_window_size is not None:
            mean_rgb = sliding_average(mean_rgb, window_size=sliding_window_size)
        
        # Highpass filter
        # mean_rgb = highpass_filter_multichannel(mean_rgb, sample_rate=30, cutoff_freq=0.1, order=5)
        
        # Bandpassed Filter
        if cutoff_freqs is not None:
            mean_rgb = bandpass_filter(mean_rgb, cutoff_freqs=cutoff_freqs)
        
        # try to plot the mean RGB values
        if plot_mean_rgb:
            plt.figure(figsize=(18, 6))
            plt.plot(mean_rgb[0], label='R')
            plt.plot(mean_rgb[1], label='G')
            plt.plot(mean_rgb[2], label='B')
            plt.xlabel('Frame'); plt.ylabel('Mean RGB'); plt.title(f'Mean RGB values of {subject}')
            plt.legend(); plt.tight_layout(); plt.savefig(os.path.join('out', f'sample_mean_rgb.png'))
            
        
        # Calculate SpO2 value
        spo2_val = calculate_spo2(mean_rgb, A=125, B=26)
        gt_val_avg = np.mean(gt_val)
        err = abs(spo2_val-gt_val_avg)
        print(f"SpO2 value for {subject}: {spo2_val} | GT: {gt_val_avg} | Error: {err}")
        
        # add the error to the list
        err_list.append(err)
        gt_list.append(gt_val_avg)
        pred_list.append(spo2_val)
        
        # break # break for the file iterator
        
    # Print the mean error
    print(f"Mean error: {np.mean(err_list)}")
    
    # plot two line graph of GT and Pred
    plt.figure(figsize=(12, 6))
    plt.plot(gt_list, label='GT', marker='o')
    plt.plot(pred_list, label='Pred', marker='o')
    plt.xlabel('Subject'); plt.ylabel('SpO2');
    plt.title(f'Comparison of GT and Predicted SpO2 values')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join('out', f'Comparison.png'))
    
    # save the gt and pred values to csv
    # make into pandas df
    df = pd.DataFrame({'SUBJECT': file_names, 'GT': gt_list, 'PRED': pred_list})
    df.to_csv(os.path.join('out', 'gt_pred_results.csv'), index=False)
    
    
    # Calculate the Pearson correlation
    corr = np.corrcoef(gt_list, pred_list)
    print(f"Pearson correlation: {corr[0, 1]}")
    

if __name__ == '__main__':
    main(
        # start_from_subject=12,
        # end_at_subject=13,
        # plot_mean_rgb=True,
        # cutoff_freqs=[0.5, 3],
        # sliding_window_size=5,
        # scenario=2,
    )