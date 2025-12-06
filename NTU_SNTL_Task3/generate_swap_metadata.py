"""
Script to generate swapped metadata (azimuth inverted) for stereo_dev_swap audio files.
This creates *_swap.csv files in metadata_dev with inverted azimuth values.
"""

import os
import pandas as pd
from tqdm import tqdm

def generate_swap_metadata(metadata_dir):
    """
    Generate swapped metadata files with inverted azimuth values.
    
    Parameters:
        metadata_dir (str): Path to the metadata_dev directory
    """
    # Get all original CSV files (not already swapped)
    csv_files = [f for f in os.listdir(metadata_dir) 
                 if f.endswith('.csv') and not f.endswith('_swap.csv')]
    
    print(f"Found {len(csv_files)} original CSV files")
    
    created_count = 0
    skipped_count = 0
    
    for csv_file in tqdm(csv_files, desc="Generating swap metadata"):
        original_path = os.path.join(metadata_dir, csv_file)
        swap_filename = csv_file.replace('.csv', '_swap.csv')
        swap_path = os.path.join(metadata_dir, swap_filename)
        
        # Skip if already exists
        if os.path.exists(swap_path):
            skipped_count += 1
            continue
        
        try:
            # Read original CSV
            df = pd.read_csv(original_path)
            
            # Invert azimuth column
            if 'azimuth' in df.columns:
                df['azimuth'] = -df['azimuth']
            else:
                print(f"Warning: 'azimuth' column not found in {csv_file}")
                continue
            
            # Save swapped metadata
            df.to_csv(swap_path, index=False)
            created_count += 1
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    print(f"\nCompleted!")
    print(f"Created: {created_count} swap metadata files")
    print(f"Skipped (already exist): {skipped_count}")


if __name__ == "__main__":
    metadata_dir = r"e:\SELD\output_2020_stereo\DCASE2020_Stereo_SELD_Dataset\metadata_dev"
    generate_swap_metadata(metadata_dir)
