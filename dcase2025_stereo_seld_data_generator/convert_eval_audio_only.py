import glob
import tqdm
import os
import librosa
import numpy as np
import soundfile as sf

def convert_foa_to_stereo(wav_path, output_path, deg):
    """
    Convert FOA audio to stereo using M/S technique
    
    Args:
        wav_path: Path to input FOA wav file
        output_path: Path to output stereo wav file
        deg: Viewing angle in degrees
    """
    sr = 24000
    
    # Load FOA audio
    audio_data, _ = librosa.load(wav_path, sr=sr, mono=False)
    
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
        return False
    
    # Save stereo audio
    sf.write(output_path, two_ch_audio_data.T, sr)
    return True


def main():
    source_dir = "./2020dataset"
    target_dir = "./output_2020_stereo"
    tag_dataset = "DCASE2020_Stereo_SELD_Dataset"
    
    # Get all eval FOA files
    foa_eval_files = sorted(glob.glob("{}/foa_eval/*.wav".format(source_dir)))
    
    print(f"Found {len(foa_eval_files)} evaluation files")
    
    # Fixed viewing angle for eval (you can change this or make it random)
    # Using 0 degrees (front view) for consistency
    deg = 0
    
    # Create output directory
    output_dir = "{}/{}/stereo_eval".format(target_dir, tag_dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    failed_files = []
    
    print(f"Converting with viewing angle: {deg} degrees")
    
    with tqdm.tqdm(total=len(foa_eval_files)) as pbar:
        for foa_path in foa_eval_files:
            # Get filename
            filename = os.path.basename(foa_path)
            base_name = filename.replace('.wav', '')
            
            # Create output path
            output_filename = f"{base_name}_deg{deg:03d}.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            # Convert
            success = convert_foa_to_stereo(foa_path, output_path, deg)
            
            if success:
                success_count += 1
            else:
                failed_files.append(filename)
            
            pbar.update(1)
    
    print(f"\nConversion complete!")
    print(f"Success: {success_count}/{len(foa_eval_files)}")
    
    if failed_files:
        print(f"Failed files (clipping detected): {len(failed_files)}")
        for f in failed_files:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
