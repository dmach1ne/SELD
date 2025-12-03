import torch
import glob
import os
from collections import defaultdict

# Check label distribution
feat_dir = '../dcase2025_stereo_seld_data_generator/output_2020_stereo/DCASE2020_Stereo_SELD_Dataset/mel64'
folds = ['fold1', 'fold2', 'fold3', 'fold4']

class_counts = defaultdict(int)
total_files = 0
total_events = 0

for fold in folds:
    label_files = glob.glob(os.path.join(feat_dir, f'metadata_dev_adpit/{fold}*.pt'))
    print(f"\n{fold}: {len(label_files)} files")
    
    for label_file in label_files[:10]:  # Check first 10 files per fold
        label = torch.load(label_file)
        # Shape: (50, 6, 5, 14) -> (frames, tracks, elements, classes)
        # Element 0 is SED (activation)
        sed = label[:, :, 0, :]  # (50, 6, 14)
        
        # Count active events per class
        for class_idx in range(14):
            active = (sed[:, :, class_idx] > 0).sum().item()
            if active > 0:
                class_counts[class_idx] += 1
                total_events += active
        
        total_files += 1

print(f"\n{'='*60}")
print(f"Total files checked: {total_files}")
print(f"Total active events: {total_events}")
print(f"\nClass distribution (number of files containing each class):")
print(f"{'Class':<10} {'Count':<10} {'Percentage':<10}")
print(f"{'-'*30}")

for class_idx in range(14):
    count = class_counts[class_idx]
    percentage = (count / total_files * 100) if total_files > 0 else 0
    print(f"{class_idx:<10} {count:<10} {percentage:.1f}%")

# Check which folds are being used for training
print(f"\n{'='*60}")
print("Checking fold usage:")
print(f"Training folds: fold1, fold2, fold3")
print(f"Test fold: fold4")

# Count files per fold
for fold in folds:
    audio_files = glob.glob(os.path.join(feat_dir, f'stereo_dev_normalized/{fold}*.pt'))
    label_files = glob.glob(os.path.join(feat_dir, f'metadata_dev_adpit/{fold}*.pt'))
    print(f"{fold}: {len(audio_files)} audio files, {len(label_files)} label files")



