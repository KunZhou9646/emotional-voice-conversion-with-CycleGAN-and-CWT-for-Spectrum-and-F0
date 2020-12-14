# Emotional Voice Conversion and/or Speaker Identity Conversion with Non-Parallel Training Data

**This is an implementation of our Cycle-GAN based emotional voice conversion framework (to be appear in Speaker Odyssey 2020) that converts both spectrum and prosody features. Please kindly cite our paper if you are using our codes:**
 
*Kun  Zhou,  Berrak  Sisman,  and  Haizhou  Li,“Transforming spectrum and prosody for emotional voice conversion with non-parallel training data,” arXiv preprint arXiv:2002.00198, 2020*

**Bibtex:**
```
@article{zhou2020transforming, 
  title={Transforming Spectrum and Prosody for Emotional Voice Conversion with Non-Parallel Training Data},
  author={Zhou, Kun and Sisman, Berrak and Li, Haizhou},
  journal={arXiv preprint arXiv:2002.00198},
  year={2020}
}

```
Dependencies
-------

  `Python 3.5`

  `Numpy 1.14`

  `Tensorflow 1.8`

  `ProgressBar2 3.37.1`

  `LibROSA 0.6`

  `FFmpeg 4.0`

  `PyWorld`

  `sklearn`

  `pycwt`

  `sprocket-vc`

 ` scipy`

  `glob`

Usage
---------

1. `train.py`

This script is to train CycleGAN with spectrum features.

2. `train_f0.py`

This script is to perform CWT on F0, then train CycleGAN with CWT-F0 features.

3. `convert_separate.py`

This script is to convert speech from the source using trained CycleGAN to convert spectrum and CWT-F0 features separately.


# Instruction

1. **To train CycleGAN with spectrum features, please run the code:**</br>
```Bash
$ python train.py --train_A_dir './data/training/NEUTRAL(PATH TO SOURCE TRAINING DATA)' --train_B_dir './data/training/SURPRISE(PATH TO TARGET TRAINING DATA)' --model_dir './model/neutral_to_suprise_mceps' --model_name 'neutral_to_suprise_mceps.ckpt' --random_seed 0 --validation_A_dir './data/evaluation_all/NEUTRAL' --validation_B_dir './data/evaluation_all/SURPRISE' --output_dir './validation_output' --tensorboard_log_dir './log'
```

2. **To train CycleGAN with CWT-F0 features, please run the code:** 
```Bash
$ python train_f0.py --train_A_dir './data/training/NEUTRAL(PATH TO SOURCE TRAINING DATA)' --train_B_dir './data/training/SURPRISE(PATH TO TARGET TRAINING DATA)' --model_dir './model/neutral_to_suprise_f0' --model_name 'neutral_to_suprise_f0.ckpt' --random_seed 0 --validation_A_dir './data/evaluation_all/NEUTRAL' --validation_B_dir './data/evaluation_all/SURPRISE' --output_dir './validation_output' --tensorboard_log_dir './log' 
```

3. **To convert the emotion from the source to the target, please run the code:**
```Bash
$ python convert_separate.py --model_f0_dir './model/neutral_to_surprise_f0' --model_f0_name 'neutral_to_surprise_f0.ckpt' --model_mceps_dir './model/neutral_to_surprise_mceps' --model_mceps_name 'neutral_to_surprise_mceps.ckpt' --data_dir './data/evaluation_all/NEUTRAL(PATH TO EVALUATION DATA)' --conversion_direction 'A2B' --output_dir './converted_voices_neutral_to_surprise_separate'
```


**Note1:** 
The codes are based on CycleGAN Voice Conversion: https://github.com/leimao/Voice_Converter_CycleGAN

**Note2:** 
The codes can easily be used for traditional parallel data free voice conversion (speaker identity conversion). You just need to change the training data to VCC2016 or VCC2018 (both publicly available) and the run the scripts. You can perform spectrum and CWT-based F0 conversion. 

