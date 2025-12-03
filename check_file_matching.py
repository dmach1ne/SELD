import os
import glob

# Check prediction files (use most recent run)
import glob as g
pred_dirs = sorted(g.glob("outputs/DCASE2020_fold6_val_audio_multiACCDOA_*/dev-test"))
if not pred_dirs:
    print("No prediction directory found!")
    exit(1)
pred_dir = pred_dirs[-1]  # Most recent
print(f"Using prediction dir: {pred_dir}")
pred_files = sorted(os.listdir(pred_dir))
print(f"Prediction files: {len(pred_files)}")
print("First 5 pred files:")
for f in pred_files[:5]:
    print(f"  {f}")

# Check reference files
ref_dir = "../dcase2025_stereo_seld_data_generator/output_2020_stereo/DCASE2020_Stereo_SELD_Dataset/metadata_dev"
ref_files = []
for split in os.listdir(ref_dir):
    split_path = os.path.join(ref_dir, split)
    if os.path.isdir(split_path):
        for f in os.listdir(split_path):
            if f.startswith('fold6'):
                ref_files.append(f)

ref_files = sorted(ref_files)
print(f"\nReference files: {len(ref_files)}")
print("First 5 ref files:")
for f in ref_files[:5]:
    print(f"  {f}")

# Check matching
print(f"\nMatching check:")
print(f"Pred files: {len(pred_files)}")
print(f"Ref files: {len(ref_files)}")
print(f"Match: {pred_files == ref_files}")

if pred_files != ref_files:
    pred_set = set(pred_files)
    ref_set = set(ref_files)
    only_in_pred = pred_set - ref_set
    only_in_ref = ref_set - pred_set
    if only_in_pred:
        print(f"\nOnly in predictions ({len(only_in_pred)}):")
        for f in list(only_in_pred)[:5]:
            print(f"  {f}")
    if only_in_ref:
        print(f"\nOnly in references ({len(only_in_ref)}):")
        for f in list(only_in_ref)[:5]:
            print(f"  {f}")

