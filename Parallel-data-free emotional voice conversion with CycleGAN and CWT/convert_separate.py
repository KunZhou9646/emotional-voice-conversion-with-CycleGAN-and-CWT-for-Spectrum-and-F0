import argparse
import os
import numpy as np

from model_f0 import CycleGAN as CycleGAN_f0
from model_mceps import CycleGAN as CycleGAN_mceps

from preprocess import *
from utils import get_lf0_cwt_norm,norm_scale,denormalize
from utils import get_cont_lf0, get_lf0_cwt,inverse_cwt
from sklearn import preprocessing

def conversion(model_f0_dir, model_f0_name, model_mceps_dir, model_mceps_name, data_dir, conversion_direction, output_dir):

    num_mceps = 24
    num_features = 34
    sampling_rate = 16000
    frame_period = 5.0

#    model = CycleGAN(num_features = num_features, mode = 'test')
#    model.load(filepath = os.path.join(model_dir, model_name))

#  import F0 model:
    model_f0 = CycleGAN_f0(num_features = 10, mode = 'test')
    model_f0.load(filepath=os.path.join(model_f0_dir,model_f0_name))
#  import mceps model:
    model_mceps = CycleGAN_mceps(num_features = 24, mode = 'test')
    model_mceps.load(filepath=os.path.join(model_mceps_dir,model_mceps_name))

    mcep_normalization_params = np.load(os.path.join(model_mceps_dir, 'mcep_normalization.npz'))
    mcep_mean_A = mcep_normalization_params['mean_A']
    mcep_std_A = mcep_normalization_params['std_A']
    mcep_mean_B = mcep_normalization_params['mean_B']
    mcep_std_B = mcep_normalization_params['std_B']

    logf0s_normalization_params = np.load(os.path.join(model_f0_dir, 'logf0s_normalization.npz'))
    logf0s_mean_A = logf0s_normalization_params['mean_A']
    logf0s_std_A = logf0s_normalization_params['std_A']
    logf0s_mean_B = logf0s_normalization_params['mean_B']
    logf0s_std_B = logf0s_normalization_params['std_B']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(data_dir):

        filepath = os.path.join(data_dir, file)
        wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
        wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
        f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
        coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_mceps)
        coded_sp_transposed = coded_sp.T
       # np.save('./f0',f0)
        uv, cont_lf0_lpf = get_cont_lf0(f0)

        if conversion_direction == 'A2B':
            #f0_converted = pitch_conversion(f0 = f0, mean_log_src = logf0s_mean_A, std_log_src = logf0s_std_A, mean_log_target = logf0s_mean_B, std_log_target = logf0s_std_B)
            #f0_converted = f0

            cont_lf0_lpf_norm = (cont_lf0_lpf - logf0s_mean_A) / logf0s_std_A
            Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_lpf_norm) #[470,10]
            #np.save('./Wavelet_lf0',Wavelet_lf0)
            Wavelet_lf0_norm, mean, std = norm_scale(Wavelet_lf0) #[470,10],[1,10],[1,10]
            lf0_cwt_norm = Wavelet_lf0_norm.T #[10,470]
            
            coded_sp_norm = (coded_sp_transposed - mcep_mean_A) / mcep_std_A #[24,470]

        #    feats_norm = np.vstack((coded_sp_norm,lf0_cwt_norm))#[34,470]
        #    feats_converted_norm = model.test(inputs = np.array([feats_norm]), direction = conversion_direction)[0]

            # test mceps
            coded_sp_converted_norm = model_mceps.test(inputs = np.array([coded_sp_norm]),direction = conversion_direction)[0]
            # test f0:
            lf0 = model_f0.test(inputs = np.array([lf0_cwt_norm]),direction=conversion_direction)[0]
            #coded_sp_converted_norm = model.test(inputs = np.array([feats_norm]), direction = conversion_direction)[0]

            #coded_sp_converted_norm = feats_converted_norm[:24]
            coded_sp_converted = coded_sp_converted_norm * mcep_std_B + mcep_mean_B #mceps

            #lf0 = feats_converted_norm[24:].T #[470,10]

            lf0_cwt_denormalize = denormalize(lf0.T, mean, std)#[470,10]
            #np.save('./lf0_denorm',lf0_cwt_denormalize)
            lf0_rec = inverse_cwt(lf0_cwt_denormalize,scales)#[470,1]
            #lf0_rec_norm = preprocessing.scale(lf0_rec)
            lf0_converted = lf0_rec * logf0s_std_B + logf0s_mean_B
            f0_converted = np.squeeze(uv) * np.exp(lf0_converted)
            f0_converted = np.ascontiguousarray(f0_converted)
            #np.save('./f0_converted',f0_converted)

        else:
            #f0_converted = pitch_conversion(f0 = f0, mean_log_src = logf0s_mean_B, std_log_src = logf0s_std_B, mean_log_target = logf0s_mean_A, std_log_target = logf0s_std_A)
            #f0_converted = f0
            cont_lf0_lpf_norm = (cont_lf0_lpf - logf0s_mean_B) / logf0s_std_B
            Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_lpf_norm)
            lf0_cwt_norm = Wavelet_lf0.T #[10,470]
            coded_sp_norm = (coded_sp_transposed - mcep_mean_B) / mcep_std_B
            feats_norm = np.vstack((coded_sp_norm,lf0_cwt_norm))#[34,470]
            feats_converted_norm = model_f0.test(inputs = np.array([feats_norm]), direction = conversion_direction)[0]

            #coded_sp_converted_norm = model.test(inputs = np.array([feats_norm]), direction = conversion_direction)[0]
            coded_sp_converted_norm = feats_converted_norm[:24]
            coded_sp_converted = coded_sp_converted_norm * mcep_std_A + mcep_mean_A
            lf0_rec = inverse_cwt(feats_norm[24:].T,scales)#[470,10]
            lf0_rec_norm = preprocessing.scale(lf0_rec)
            lf0_converted = lf0_rec_norm * logf0s_std_A + logf0s_mean_A
            f0_converted = np.squeeze(uv) * np.exp(lf0_converted)
            f0_converted = np.ascontiguousarray(f0_converted)

            #coded_sp_norm = (coded_sp_transposed - mcep_mean_B) / mcep_std_B
            #coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]), direction = conversion_direction)[0]
            #coded_sp_converted = coded_sp_converted_norm * mcep_std_A + mcep_mean_A

        coded_sp_converted = coded_sp_converted.T#[470,24]
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
        wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
        librosa.output.write_wav(os.path.join(output_dir, os.path.basename(file)), wav_transformed, sampling_rate)


if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES']='2'
    parser = argparse.ArgumentParser(description = 'Convert voices using pre-trained CycleGAN model.')

    model_f0_dir_default = './model/neutral_to_surprise_f0'
    model_f0_name_default = 'neutral_to_surprise_f0.ckpt'
    model_mceps_dir_default = './model/neutral_to_surprise_mceps'
    model_mceps_name_default = 'neutral_to_surprise_mceps.ckpt'
    data_dir_default = './data/evaluation_all/NEUTRAL'
    conversion_direction_default = 'A2B'
    output_dir_default = './converted_voices_neutral_to_surprise_separate'

    parser.add_argument('--model_f0_dir', type = str, help = 'Directory for the pre-trained f0 model.', default = model_f0_dir_default)
    parser.add_argument('--model_f0_name', type = str, help = 'Filename for the pre-trained f0 model.', default = model_f0_name_default)
    parser.add_argument('--model_mceps_dir', type = str, help = 'Directory for the pre-trained mceps model.', default = model_mceps_dir_default)
    parser.add_argument('--model_mceps_name', type = str, help = 'Filename for the pre-trained mceps model.', default = model_mceps_name_default)
    parser.add_argument('--data_dir', type = str, help = 'Directory for the voices for conversion.', default = data_dir_default)
    parser.add_argument('--conversion_direction', type = str, help = 'Conversion direction for CycleGAN. A2B or B2A. The first object in the model file name is A, and the second object in the model file name is B.', default = conversion_direction_default)
    parser.add_argument('--output_dir', type = str, help = 'Directory for the converted voices.', default = output_dir_default)

    argv = parser.parse_args()

    model_f0_dir = argv.model_f0_dir
    model_f0_name = argv.model_f0_name
    model_mceps_dir = argv.model_mceps_dir
    model_mceps_name = argv.model_mceps_name
    data_dir = argv.data_dir
    conversion_direction = argv.conversion_direction
    output_dir = argv.output_dir

    conversion(model_f0_dir = model_f0_dir, model_f0_name = model_f0_name, model_mceps_dir = model_mceps_dir, model_mceps_name = model_mceps_name, data_dir = data_dir, conversion_direction = conversion_direction, output_dir = output_dir)


