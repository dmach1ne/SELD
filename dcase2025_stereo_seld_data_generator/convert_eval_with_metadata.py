import glob
import tqdm
import pandas as pd
import os
import librosa
import numpy as np
import soundfile as sf

from utils import CheckOnOffFast, set_range_minus180to180, fold_back_azimuth


def convert_eval_with_metadata(source_dir, target_dir, tag_dataset, deg=0):
    """
    Convert evaluation FOA audio to stereo with metadata
    
    Args:
        source_dir: Source directory containing foa_eval and metadata_eval
        target_dir: Target directory for output
        tag_dataset: Dataset tag name
        deg: Fixed viewing angle (default 0 for front view)
    """
    sr = 24000
    
    # Get all eval files
    foa_eval_files = sorted(glob.glob("{}/foa_eval/*.wav".format(source_dir)))
    metadata_eval_files = sorted(glob.glob("{}/metadata_eval/metadata_eval/*.csv".format(source_dir)))
    
    print(f"Found {len(foa_eval_files)} audio files")
    print(f"Found {len(metadata_eval_files)} metadata files")
    
    # Create output directories
    audio_output_dir = "{}/{}/stereo_eval".format(target_dir, tag_dataset)
    metadata_output_dir = "{}/{}/metadata_eval".format(target_dir, tag_dataset)
    os.makedirs(audio_output_dir, exist_ok=True)
    os.makedirs(metadata_output_dir, exist_ok=True)
    
    check_on_off_fast = CheckOnOffFast()
    
    success_count = 0
    failed_files = []
    
    print(f"Converting with viewing angle: {deg} degrees")
    
    with tqdm.tqdm(total=len(foa_eval_files)) as pbar:
        for foa_path in foa_eval_files:
            # Get filename
            filename = os.path.basename(foa_path)
            base_name = filename.replace('.wav', '')
            
            # Find corresponding metadata
            metadata_path = "{}/metadata_eval/metadata_eval/{}.csv".format(source_dir, base_name)
            
            if not os.path.exists(metadata_path):
                print(f"Warning: Metadata not found for {base_name}")
                failed_files.append(filename)
                pbar.update(1)
                continue
            
            # Load FOA audio
            audio_data, _ = librosa.load(foa_path, sr=sr, mono=False)
            
            # Extract FOA channels
            audio_data_W = audio_data[0, :]
            audio_data_Y = audio_data[1, :]
            audio_data_Z = audio_data[2, :]
            audio_data_X = audio_data[3, :]
            
            # Rotate based on viewing angle
            audio_len = len(audio_data_W)
            deg_list = np.ones(audio_len) * deg
            rad_list = deg_list / 180 * np.pi
            
            audio_data_newX = np.multiply(np.cos(rad_list), audio_data_X) - np.multiply(np.sin(rad_list), audio_data_Y)
            audio_data_newY = np.multiply(np.sin(rad_list), audio_data_X) + np.multiply(np.cos(rad_list), audio_data_Y)
            
            # Convert to stereo using M/S technique
            two_ch_audio_data = np.zeros((2, audio_len))
            two_ch_audio_data[0, :] = audio_data_W + audio_data_newY  # left
            two_ch_audio_data[1, :] = audio_data_W - audio_data_newY  # right
            
            # Check for clipping
            if np.max(np.abs(two_ch_audio_data)) > 0.9999:
                failed_files.append(filename)
                pbar.update(1)
                continue
            
            # Save stereo audio
            output_audio_path = os.path.join(audio_output_dir, f"{base_name}_deg{deg:03d}.wav")
            sf.write(output_audio_path, two_ch_audio_data.T, sr)
            
            # Process metadata
            metadata_df = pd.read_csv(metadata_path, sep=",", header=None)
            
            cut_df_perspective = pd.DataFrame(columns=["frame", "class", "source", "azimuth", "elevation", "distance", "onscreen", "x_on", "y_on"])
            
            for row in metadata_df.values:
                # Handle both 2020 format (5 cols) and 2023+ format (6 cols)
                if len(row) == 5:
                    frame, category, source, azi, ele = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])
                    dist = 1  # Default distance for 2020 dataset
                else:
                    frame, category, source, azi, ele, dist = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5])
                
                is_onscreen, x_on, y_on = check_on_off_fast.check(deg, azi, ele)
                cut_df_perspective.loc[len(cut_df_perspective.index)] = [
                    frame, category, source,
                    fold_back_azimuth(set_range_minus180to180(azi + deg)), ele, dist,
                    is_onscreen, x_on, y_on
                ]
            
            # Save metadata
            output_metadata_path = os.path.join(metadata_output_dir, f"{base_name}_deg{deg:03d}.csv")
            cut_df_perspective_simple = cut_df_perspective[["frame", "class", "source", "azimuth", "distance", "onscreen"]]
            cut_df_perspective_simple_int = cut_df_perspective_simple.astype(int)
            cut_df_perspective_simple_int.to_csv(output_metadata_path, sep=',', index=False, header=True)
            
            success_count += 1
            pbar.update(1)
    
    print(f"\nConversion complete!")
    print(f"Success: {success_count}/{len(foa_eval_files)}")
    
    if failed_files:
        print(f"Failed files: {len(failed_files)}")
        for f in failed_files:
            print(f"  - {f}")


if __name__ == "__main__":
    source_dir = "./2020dataset"
    target_dir = "./output_2020_stereo"
    tag_dataset = "DCASE2020_Stereo_SELD_Dataset"
    
    # Fixed viewing angle for eval (0 degrees = front view)
    deg = 0
    
    convert_eval_with_metadata(source_dir, target_dir, tag_dataset, deg)
