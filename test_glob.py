import glob
import os

feat_dir = 'dcase2025_stereo_seld_data_generator/output_2020_stereo/DCASE2020_Stereo_SELD_Dataset/mel64'
fold = 'fold1'

# Test audio pattern
audio_pattern = os.path.join(feat_dir, f'stereo_dev_normalized/{fold}*.pt')
audio_files = glob.glob(audio_pattern)
print(f"Audio pattern: {audio_pattern}")
print(f"Found {len(audio_files)} audio files")
if audio_files:
    print(f"First audio file: {audio_files[0]}")

# Test label pattern
label_pattern = os.path.join(feat_dir, f'metadata_dev_adpit/{fold}*.pt')
label_files = glob.glob(label_pattern)
print(f"\nLabel pattern: {label_pattern}")
print(f"Found {len(label_files)} label files")
if label_files:
    print(f"First label file: {label_files[0]}")

# Check if directories exist
stereo_dir = os.path.join(feat_dir, 'stereo_dev_normalized')
metadata_dir = os.path.join(feat_dir, 'metadata_dev_adpit')
print(f"\nStereo dir exists: {os.path.exists(stereo_dir)}")
print(f"Metadata dir exists: {os.path.exists(metadata_dir)}")

# List first few files in stereo_dev_normalized
if os.path.exists(stereo_dir):
    files = os.listdir(stereo_dir)[:5]
    print(f"\nFirst 5 files in stereo_dev_normalized:")
    for f in files:
        print(f"  {f}")



