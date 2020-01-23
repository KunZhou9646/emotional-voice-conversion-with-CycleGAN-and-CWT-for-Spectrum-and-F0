import os
import numpy as np
import argparse
import librosa
from sklearn import preprocessing
from preprocess import *
import sprocket
from sprocket.util import melcd, estimate_twf
from utils import convert_continuos_f0
from scipy.stats.stats import pearsonr
import glob

def evaluation(source_data_dir, target_data_dir):

    sampling_rate = 16000
    frame_period = 5.0
    num_features = 24

    f0_source = list()
    mceps_source = list()
    f0_target = list()
    mceps_target = list()

    source_path_list = os.listdir(source_data_dir)
    source_path_list.sort()
    target_path_list = os.listdir(target_data_dir)
    target_path_list.sort()

    for file_s in source_path_list:

        filepath_s = os.path.join(source_data_dir, file_s)
        wav, _ = librosa.load(filepath_s, sr = sampling_rate, mono = True)
        wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
        f0_s, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
        f0_source.append(f0_s) #[640,]
        coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_features)
        mceps_source.append(coded_sp) #[640,24]
        #print(file_s)

    for file_t in target_path_list:

        filepath_t = os.path.join(target_data_dir, file_t)
        wav, _ = librosa.load(filepath_t, sr = sampling_rate, mono = True)
        wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
        f0_t, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
        f0_target.append(f0_t)
        coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_features)
        mceps_target.append(coded_sp)
        #print(file_t)

    # Calculate PCC:
    print("Calculating PCC")
    PCC = 0
    PCC_sum = 0
    PCC_average = 0
    for f0_s, f0_t in zip(f0_source, f0_target):
        uv_s, cont_f0_s = convert_continuos_f0(f0_s)
        uv_t, cont_f0_t = convert_continuos_f0(f0_t)
        cont_f0_s = cont_f0_s[...,None]
        cont_f0_t = cont_f0_t[...,None]
        # dtw
        twf = estimate_twf(cont_f0_s, cont_f0_t, fast=True)
        orgmod = cont_f0_s[twf[0]]
        tarmod = cont_f0_t[twf[1]]
        assert orgmod.shape == tarmod.shape
        PCC,_ = pearsonr(orgmod, tarmod)
        PCC_sum += PCC
        print(PCC)
    PCC_average = PCC_sum / 10
    print("Average PCC is", PCC_average)

# Calculating MCD:
    print("Calculating MCD")
    mcd = 0
    mcd_sum = 0
    for mceps_s,mceps_t in zip(mceps_source,mceps_target):
        #dtw
        def distance_func(x, y): return melcd(x, y)
        mceps_s = mceps_s[:,1:]
        mceps_t = mceps_t[:,1:]
        twf = estimate_twf(mceps_s, mceps_t, fast=True)
        orgmod = mceps_s[twf[0]]
        tarmod = mceps_t[twf[1]]
        assert orgmod.shape == tarmod.shape
        mcd = melcd(orgmod,tarmod)
        mcd_sum += mcd
        print(mcd)
    mcd_average = mcd_sum / 10
    print("Average MCD is", mcd_average)

#Calculating RMSE:
    print("Calculating RMSE")
    RMSE = 0
    RMSE_sum = 0
    RMSE_average = 0
    for f0_s,f0_t in zip(f0_source,f0_target):

        #uv_s, cont_f0_s = convert_continuos_f0(f0_s)
        #uv_t, cont_f0_t = convert_continuos_f0(f0_t)
        #log_f0_s = np.log(cont_f0_s) * np.squeeze(uv_s)
        #log_f0_t = np.log(cont_f0_t) * np.squeeze(uv_t)
        #log_f0_s = log_f0_s[..., None]
        #log_f0_t = log_f0_t[..., None]

        f0_s = f0_s[..., None]
        f0_t = f0_t[..., None]

        #dtw
        #twf = estimate_twf(log_f0_s, log_f0_t, fast=True)
        twf = estimate_twf(f0_s, f0_t, fast=True)
        orgmod = f0_s[twf[0]]
        tarmod = f0_t[twf[1]]
        assert orgmod.shape == tarmod.shape
        diff = orgmod - tarmod
        RMSE = np.sqrt(np.mean(diff**2))
        RMSE_sum += RMSE
        print(RMSE)
    RMSE_average = RMSE_sum / 10
    print("Average RMSE is", RMSE_average)










if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Evaluation.')

    source_data_dir_default = './Results/Baseline2/converted_voices_10cwt_anger'
    target_data_dir_default = './Results/References/ANGER'

    parser.add_argument('--source_data_dir', type = str, help = 'Directory for the source data.', default = source_data_dir_default)
    parser.add_argument('--target_data_dir', type = str, help = 'Directory for the target data.', default = target_data_dir_default)

    argv = parser.parse_args()

    source_data_dir = argv.source_data_dir
    target_data_dir = argv.target_data_dir


    evaluation(source_data_dir = source_data_dir, target_data_dir = target_data_dir)
