import torch
import sys

label_path = '../dcase2025_stereo_seld_data_generator/output_2020_stereo/DCASE2020_Stereo_SELD_Dataset/mel64/metadata_dev_adpit/fold1_room1_mix001_ov1_deg215_start0056.pt'

try:
    label = torch.load(label_path)
    print(f"Label shape: {label.shape}")
    print(f"Label dtype: {label.dtype}")
    print(f"Label min/max: {label.min():.4f} / {label.max():.4f}")
    print(f"Non-zero elements: {(label != 0).sum().item()} / {label.numel()}")
    
    # Check structure for multiACCDOA
    # Expected: (frames, 6, 5, classes) -> (50, 6, 5, 14)
    # 6 tracks, 5 elements (sed, x, y, dist, onscreen)
    if label.ndim == 4:
        print(f"\nMultiACCDOA format detected")
        print(f"Frames: {label.shape[0]}")
        print(f"Tracks: {label.shape[1]}")
        print(f"Elements: {label.shape[2]}")
        print(f"Classes: {label.shape[3]}")
        
        # Check SED (first element)
        sed = label[:, :, 0, :]
        print(f"\nSED (activation) stats:")
        print(f"  Active frames: {(sed > 0).any(dim=(1,2)).sum().item()} / {label.shape[0]}")
        print(f"  Total active events: {(sed > 0).sum().item()}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()



