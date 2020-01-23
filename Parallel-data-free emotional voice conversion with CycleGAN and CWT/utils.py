import tensorflow as tf
import os
import random
import numpy as np
from scipy.interpolate import interp1d
import pycwt as wavelet
from scipy.signal import firwin
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pywt
import librosa
import pyworld

def l1_loss(y, y_hat):

    return tf.reduce_mean(tf.abs(y - y_hat))

def l2_loss(y, y_hat):

    return tf.reduce_mean(tf.square(y - y_hat))

def cross_entropy_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))


def convert_continuos_f0(f0):
    """CONVERT F0 TO CONTINUOUS F0

    Args:
        f0 (ndarray): original f0 sequence with the shape (T)

    Return:
        (ndarray): continuous f0 with the shape (T)
    """
    # get uv information as binary
    uv = np.float32(f0 != 0)

    # get start and end of f0
    if (f0 == 0).all():
        logging.warn("all of the f0 values are 0.")
        return uv, f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nz_frames = np.where(f0 != 0)[0]

    # perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return uv, cont_f0


def get_cont_lf0(f0, frame_period=5.0):
    uv, cont_f0_lpf = convert_continuos_f0(f0)
    #cont_f0_lpf = low_pass_filter(cont_f0_lpf, int(1.0 / (frame_period * 0.001)), cutoff=20)
    cont_lf0_lpf = np.log(cont_f0_lpf)
    return uv, cont_lf0_lpf

#def get_log_energy(sp):
    # sp: (T, D)
    # return: (T)
#    energy = np.linalg.norm(sp, ord=2, axis=-1)
#    return np.log(energy)

def get_lf0_cwt(lf0):
    mother = wavelet.MexicanHat()
    #dt = 0.005
    dt = 0.005
    dj = 1
    s0 = dt*2
    J =9
    #C_delta = 3.541
    #Wavelet_lf0, scales, _, _, _, _ = wavelet.cwt(np.squeeze(lf0), dt, dj, s0, J, mother)
    Wavelet_lf0, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(np.squeeze(lf0), dt, dj, s0, J, mother)
   #Wavelet_le, scales, _, _, _, _ = wavelet.cwt(np.squeeze(le), dt, dj, s0, J, mother)
    Wavelet_lf0 = np.real(Wavelet_lf0).T
   #Wavelet_le = np.real(Wavelet_le).T   # (T, D=10)
  #0lf0_le_cwt = np.concatenate((Wavelet_lf0, Wavelet_le), -1)
  #  iwave = wavelet.icwt(np.squeeze(lf0), scales, dt, dj, mother) * std
    return Wavelet_lf0, scales


def inverse_cwt(Wavelet_lf0,scales):
    lf0_rec = np.zeros([Wavelet_lf0.shape[0],len(scales)])
    for i in range(0,len(scales)):
        lf0_rec[:,i] = Wavelet_lf0[:,i]*((i+1+2.5)**(-2.5))
    lf0_rec_sum = np.sum(lf0_rec,axis = 1)
    lf0_rec_sum = preprocessing.scale(lf0_rec_sum)
    return lf0_rec_sum


def low_pass_filter(x, fs, cutoff=70, padding=True):
    """FUNCTION TO APPLY LOW PASS FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low pass filter

    Return:
        (ndarray): Low pass filtered waveform sequence
    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    numtaps = 255
    fil = firwin(numtaps, norm_cutoff)
    x_pad = np.pad(x, (numtaps, numtaps), 'edge')
    lpf_x = lfilter(fil, 1, x_pad)
    lpf_x = lpf_x[numtaps + numtaps // 2: -numtaps // 2]
    return lpf_x

def norm_scale(Wavelet_lf0):
    Wavelet_lf0_norm = np.zeros((Wavelet_lf0.shape[0], Wavelet_lf0.shape[1]))
    mean = np.zeros((1,Wavelet_lf0.shape[1]))#[1,10]
    std = np.zeros((1, Wavelet_lf0.shape[1]))
    for scale in range(Wavelet_lf0.shape[1]):
        mean[:,scale] = Wavelet_lf0[:,scale].mean()
        std[:,scale] = Wavelet_lf0[:,scale].std()
        Wavelet_lf0_norm[:,scale] = (Wavelet_lf0[:,scale]-mean[:,scale])/std[:,scale]
    return Wavelet_lf0_norm, mean, std

def get_lf0_cwt_norm(f0s, mean, std):

    uvs = list()
    cont_lf0_lpfs = list()
    cont_lf0_lpf_norms = list()
    Wavelet_lf0s = list()
    Wavelet_lf0s_norm = list()
    scaless = list()

    means = list()
    stds = list()
    for f0 in f0s:

        uv, cont_lf0_lpf = get_cont_lf0(f0)
        cont_lf0_lpf_norm = (cont_lf0_lpf - mean) / std 

        Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_lpf_norm) #[560,10]
        Wavelet_lf0_norm, mean_scale, std_scale = norm_scale(Wavelet_lf0) #[560,10],[1,10],[1,10]

        Wavelet_lf0s_norm.append(Wavelet_lf0_norm)
        uvs.append(uv)
        cont_lf0_lpfs.append(cont_lf0_lpf)
        cont_lf0_lpf_norms.append(cont_lf0_lpf_norm)
        Wavelet_lf0s.append(Wavelet_lf0)
        scaless.append(scales)
        means.append(mean_scale)
        stds.append(std_scale)

    return Wavelet_lf0s_norm,scaless, means, stds


def denormalize(Wavelet_lf0_norm, mean, std):
    Wavelet_lf0_denorm = np.zeros((Wavelet_lf0_norm.shape[0], Wavelet_lf0_norm.shape[1]))
    for scale in range(Wavelet_lf0_norm.shape[1]):
        Wavelet_lf0_denorm[:,scale] = Wavelet_lf0_norm[:,scale]*std[:,scale]+mean[:,scale]
    return Wavelet_lf0_denorm

