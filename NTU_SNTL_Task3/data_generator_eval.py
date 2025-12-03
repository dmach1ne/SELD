"""
Special data generator for eval set that handles 60-second audio
by using sliding windows of 5 seconds
"""
import torch
from torch.utils.data.dataset import Dataset
import glob
import os


class EvalDataGenerator(Dataset):
    def __init__(self, params):
        """
        Eval data generator that splits 60-sec audio into 5-sec chunks
        """
        self.params = params
        self.feat_dir = params['feat_dir']
        self.window_size = 50  # 5 seconds = 50 frames
        
        # Get eval audio and label files
        self.audio_files = sorted(glob.glob(os.path.join(self.feat_dir, 'stereo_eval_normalized/*.pt')))
        self.label_files_orig = sorted(glob.glob(os.path.join(self.feat_dir, 'metadata_eval_adpit/*.pt')))
        
        # Create expanded label_files list for compatibility with val_epoch
        # Each file is repeated 12 times (one for each window)
        self.label_files = []
        for label_file in self.label_files_orig:
            for window_idx in range(12):
                # Create a simple filename that will work with write_logits_to_dcase_format
                # It expects to extract basename and replace .pt with .csv
                base_name = os.path.basename(label_file)[:-3]  # e.g., "mix001_deg000"
                window_name = f"{base_name}_w{window_idx:02d}.pt"  # e.g., "mix001_deg000_w00.pt"
                self.label_files.append(window_name)
        
        print(f"Eval set: {len(self.audio_files)} files × 12 windows = {len(self.label_files)} samples")
        
    def __len__(self):
        # Each 60-sec file can be split into multiple 5-sec windows
        # For simplicity, use non-overlapping windows: 60/5 = 12 windows per file
        return len(self.audio_files) * 12
    
    def __getitem__(self, item):
        """
        Returns 5-second chunks from 60-second eval files
        """
        file_idx = item // 12  # Which file
        window_idx = item % 12  # Which 5-sec window (0-11)
        
        # Load full 60-sec audio and label
        audio_file = self.audio_files[file_idx]
        label_file = self.label_files[file_idx]
        
        audio_features = torch.load(audio_file)  # (2, 4801, 64)
        labels = torch.load(label_file)  # (600, 6, 5, 14)
        
        # Extract 5-second window
        start_frame = window_idx * 50
        end_frame = min(start_frame + 50, labels.shape[0])  # Don't exceed label length
        
        # Make sure we have exactly 50 frames
        if end_frame - start_frame < 50:
            # If last window is shorter, move start back
            end_frame = labels.shape[0]
            start_frame = max(0, end_frame - 50)
        
        # Audio: need to calculate corresponding time steps
        # Ensure we get exactly 401 time steps (same as dev set)
        audio_total_frames = labels.shape[0]
        audio_total_timesteps = audio_features.shape[1]
        
        # Calculate proportional start position
        start_time = int(start_frame * audio_total_timesteps / audio_total_frames)
        
        # Ensure we don't exceed audio length
        if start_time + 401 > audio_total_timesteps:
            start_time = audio_total_timesteps - 401
        
        audio_chunk = audio_features[:, start_time:start_time+401, :]  # (2, 401, 64) - fixed size
        label_chunk = labels[start_frame:start_frame+50, :, :, :]  # (50, 6, 5, 14) - fixed size
        
        # Process labels same as training
        if not self.params['multiACCDOA']:
            mask = label_chunk[:, :self.params['nb_classes']]
            mask = mask.repeat(1, 4)
            label_chunk = mask * label_chunk[:, self.params['nb_classes']:]
        
        # Remove on/off labels for audio only task
        if self.params['multiACCDOA']:
            label_chunk = label_chunk[:, :, :-1, :]  # (50, 6, 4, 14)
        else:
            label_chunk = label_chunk[:, :-self.params['nb_classes']]
        
        return audio_chunk, label_chunk

