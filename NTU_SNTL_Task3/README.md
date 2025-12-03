# Improved Feature Engineering and Data Augmentation for 3-Dimensional Sound Event Localization and Detection with Distance Estimation

This repository implements the audio-only system of our DCASE 2025 Challenge submission. 

Within this work, we propose a set of perceptually-motivated input features, including Mid-Side (MS) spectrograms, Mid-Side Intensity Vector (IV), and the Magnitude-Squared Coherence (MSC) between the stereo channels. 

In addition, we also introduce the SED-based FilterAugment augmentation method for the SELD task. We also take Time-Frequency Masking (TFM) techniques and adapt them for the stereo-based SELD task.  

## Citation

If you have any questions, please reach out to us at: `junwei004@e.ntu.edu.sg`

If you found this code useful for your research, please consider citing our papers. 

Our [DCASE Challenge Technical Report](https://dcase.community/documents/challenge2025/technical_reports/DCASE2025_Yeow_10_t3.pdf)
```
@techreport{Yeow_NTU_task3a_report,
    Author = "Yeow, Jun-Wei and Tan, Ee-Leng and Peksi, Santi and Gan, Woon-Seng",
    title = "IMPROVING STEREO 3D SOUND EVENT LOCALIZATION AND DETECTION: PERCEPTUAL FEATURES, STEREO-SPECIFIC DATA AUGMENTATION, AND DISTANCE NORMALIZATION",
    institution = "DCASE2025 Challenge",
    year = "2025"
}
```

## Setup

This setup has been tested using Python 3.9.16 and Torch 1.13.1

```
conda create --name dcase2025_task3 python=3.9.16
conda activate dcase2025_task3
pip install -r requirements.txt
```

## Dataset

This work uses the stereo version of the Sony TAu Realistic Soundscapes and Scenes 2023 (STARSS23) dataset which can be found [here](https://zenodo.org/records/15559774).

Optionally, you should also consider creating additional First Order Ambisonics (FOA) data to help mitigate the class imbalance of the STARSS23 dataset. We recommend using [SpatialScaper](https://github.com/iranroman/SpatialScaper) for this. After generating the FOA audio, convert them into stereo signals using the official DCASE [Stereo SELD Data Generator](https://github.com/SonyResearch/dcase2025_stereo_seld_data_generator).

Store all the relevant data into an overall dataset folder, named `DCASE2025_SELD_dataset` for our project.

In our challenge submission, we applied channel swapping onto the real recordings. You can use `left_right_swap.py` for this purpose. 

## Getting started

We use a single script `main.py` to run all experiments. Within `main.py`, an argument parser is used to control relevant experiment parameters. This script will extract features, perform feature normalization, and train the desired SELD model. Note that a unique folder will be used to store each feature combination and labels. 

For example:

Training the baseline SELDNet with our proposed Distance Normalization method (enabled by default)
```
python3 main.py
```

Adding the MS log-Mel spectrograms, IV, and MSC (denoted as gamma within the paper)
```
python3 main.py --ms --iv --gamma
```

Enabling our proposed Inter-channel Level-Aware Time-Frequency Masking (ITFM)
```
python3 main.py --itfm
```

Please feel free to add/remove the arguments as necessary. 

## Augmentation Methods

We provide all data augmentation techniques used for this work in `training_utils.py`. These methods can be exported and used for other work. 

Do note that these augmentations are designed to be applied to data samples of `torch.tensor`. They accept both 3D and 4D tensors of shape `(batch), channel, time, frequency`. The `FilterAugment` augmentation was originally designed for power spectrograms (in dB). We provide a more generalized version that works with our normalized feature maps in `FilterAugmentNormalized`. 

### Time-Frequency Masking (TFM)

Three types of TFM methods are used, and subsequently combined into a single function `CompositeCutoutTorch`. To swap to the Inter-channel Level-Aware TFM (I-TFM) variants, just set the `use_itfm` flag to be `True`. The individual methods are:

- **RandomCutout** (One large rectangular mask)
- **RandomCutoutHole** (Many smaller rectangular masks)
- **SpecAugment** (Masks out entire time/frequency bands)


### Frequency Manipulation (FQM)

In our work, we experiment with Frequency Shifting and FilterAugment. To call each individual function please use:

- **RandomShiftUpDownTorch** (Frequency Shifting)
- **FilterAugmentNormalized** (FilterAugment but generalized to all scales)

## References and Acknowledgement

This repository is based on the [DCASE 2025 SELD Baseline](https://github.com/partha2409/DCASE2025_seld_baseline).


The STARSS23 dataset used for this work:
```
@article{shimada2024starss23,
  title={STARSS23: An audio-visual dataset of spatial recordings of real scenes with spatiotemporal annotations of sound events},
  author={Shimada, Kazuki and Politis, Archontis and Sudarsanam, Parthasaarathy and Krause, Daniel A and Uchida, Kengo and Adavanne, Sharath and Hakala, Aapo and Koyama, Yuichiro and Takahashi, Naoya and Takahashi, Shusuke and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

The SpatialScaper generator used to synthesize additional FOA recordings:
```
@inproceedings{Roman2024_SpatialScaper,
    Author = "Roman, Iran R. and Ick, Christopher and Ding, Sivan and Roman, Adrian S. and McFee, Brian and Bello, Juan P.",
    title = "Spatial Scaper: A Library to Simulate and Augment Soundscapes for Sound Event Localization and Detection in Realistic Rooms",
    year = "2024",
    address = "Seoul, South Korea",
    booktitle = "IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)"
}
```

The original implementation of FilterAugment was proposed in:
```
@inproceedings{nam2022filteraugment,
  title={Filteraugment: An acoustic environmental data augmentation method},
  author={Nam, Hyeonuk and Kim, Seong-Hu and Park, Yong-Hwa},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={4308--4312},
  year={2022},
  organization={IEEE}
}
```
