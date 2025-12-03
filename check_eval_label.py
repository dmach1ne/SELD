import torch
import glob
import os

feat_dir = '../dcase2025_stereo_seld_data_generator/output_2020_stereo/DCASE2020_Stereo_SELD_Dataset/mel64'

# Check eval label
eval_label_files = glob.glob(os.path.join(feat_dir, 'metadata_eval_adpit/*.pt'))
print(f"Found {len(eval_label_files)} eval label files")

if eval_label_files:
    label = torch.load(eval_label_files[0])
    print(f"\nEval label shape: {label.shape}")
    print(f"Eval label dtype: {label.dtype}")
    
# Check dev label for comparison
dev_label_files = glob.glob(os.path.join(feat_dir, 'metadata_dev_adpit/fold1*.pt'))
if dev_label_files:
    label = torch.load(dev_label_files[0])
    print(f"\nDev label shape: {label.shape}")
    print(f"Dev label dtype: {label.dtype}")

# Check eval audio
eval_audio_files = glob.glob(os.path.join(feat_dir, 'stereo_eval_normalized/*.pt'))
if eval_audio_files:
    audio = torch.load(eval_audio_files[0])
    print(f"\nEval audio shape: {audio.shape}")
    
# Check dev audio for comparison  
dev_audio_files = glob.glob(os.path.join(feat_dir, 'stereo_dev_normalized/fold1*.pt'))
if dev_audio_files:
    audio = torch.load(dev_audio_files[0])
    print(f"Dev audio shape: {audio.shape}")



