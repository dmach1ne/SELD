"""
realtime_inference.py

Real-time SELD inference with microphone input.
Captures stereo audio, extracts features, and displays sound source directions.

Usage:
    python realtime_inference.py                    # Live microphone input
    python realtime_inference.py --file test.wav    # Test with file
    python realtime_inference.py --list-devices     # List audio devices

Requirements:
    pip install pyaudio
"""

import os
import sys
import time
import argparse
import pickle
import threading
import queue
import numpy as np
import torch
import joblib
from parameters import params
from model import SELDModel
import utils

# SELD class names (DCASE2020)
CLASS_NAMES = [
    'alarm', 'baby', 'crash', 'dog', 'engine', 'female_scream',
    'female_speech', 'fire', 'footsteps', 'knock', 'male_scream',
    'male_speech', 'phone', 'piano'
]


class RealtimeSELD:
    def __init__(self, checkpoint_dir, device='cuda', slide_interval_ms=500):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.slide_interval_ms = slide_interval_ms
        
        # Load config from checkpoint
        config_path = os.path.join(checkpoint_dir, 'config.pkl')
        with open(config_path, 'rb') as f:
            self.params = pickle.load(f)
        print(f"Loaded config from: {config_path}")
        
        # Audio parameters
        self.sr = self.params['sampling_rate']
        self.hop_length = int(self.sr * self.params['hop_length_s'])
        self.n_fft = 2 ** (2 * self.hop_length - 1).bit_length()
        self.window_samples = int(5.0 * self.sr)  # 5 seconds
        
        # Calculate input channels
        n_channels = 2  # stereo mel
        if self.params.get('ms'): n_channels += 2
        if self.params.get('gamma'): n_channels += 1
        if self.params.get('ipd'): n_channels += 2
        if self.params.get('iv'): n_channels += 1
        if self.params.get('slite'): n_channels += 1
        
        # Load model
        model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        in_feat_shape = (1, n_channels, 251, self.params['nb_mels'])
        self.model = SELDModel(self.params, in_feat_shape)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['seld_model'])
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded: {model_path}")
        
        # Load scaler
        scaler_path = os.path.join(self.params['feat_dir'], 'scaler_dev.pkl')
        if not os.path.exists(scaler_path):
            # Search in parent directory
            parent_dir = os.path.dirname(self.params['feat_dir'])
            for folder in os.listdir(parent_dir):
                candidate = os.path.join(parent_dir, folder, 'scaler_dev.pkl')
                if os.path.exists(candidate):
                    scaler_path = candidate
                    break
        self.scalers = joblib.load(scaler_path)
        print(f"Scaler loaded: {scaler_path}")
        
        # Audio buffer
        self.audio_buffer = np.zeros((2, self.window_samples), dtype=np.float32)
        self.buffer_lock = threading.Lock()
        
        # Results queue
        self.results_queue = queue.Queue()
        self.running = False
        
        print(f"\nConfiguration:")
        print(f"  Sample rate: {self.sr} Hz")
        print(f"  Window: 5 seconds ({self.window_samples} samples)")
        print(f"  Slide interval: {slide_interval_ms} ms")
        print(f"  Device: {self.device}")
    
    def extract_features(self, audio):
        """Extract features from audio buffer."""
        features = utils.extract_stereo_features(
            audio, sr=self.sr, n_fft=self.n_fft, 
            hop_length=self.hop_length, win_length=self.n_fft,
            nb_mels=self.params['nb_mels'], max_freq=self.params['max_freq'],
            use_gamma=self.params['gamma'], use_ipd=self.params['ipd'],
            use_iv=self.params['iv'], use_slite=self.params['slite'], 
            use_ms=self.params['ms']
        )
        
        # Normalize
        for ch, scaler in enumerate(self.scalers):
            channel_data = features[ch].reshape(-1, features.shape[2])
            features[ch] = scaler.transform(channel_data).reshape(features.shape[1], features.shape[2])
        
        return features
    
    def infer(self, features):
        """Run model inference."""
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(feat_tensor)
        
        # Parse multi-ACCDOA output
        result = utils.get_multiaccdoa_labels(
            output, self.params['nb_classes'], 
            dnorm=self.params['dnorm'], params=self.params
        )
        return result
    
    def parse_results(self, result, debug=False):
        """Parse model output to human-readable format."""
        sed0, _, doa0, dist0, _, sed1, _, doa1, dist1, _, sed2, _, doa2, dist2, _ = result
        
        detections = []
        for slot, (sed, doa, dist) in enumerate([(sed0, doa0, dist0), (sed1, doa1, dist1), (sed2, doa2, dist2)]):
            sed_np = sed.cpu().numpy()[0]  # (50, 14)
            doa_np = doa.cpu().numpy()[0]  # (50, 28)
            
            if debug and slot == 0:
                # Show detection probabilities for last frame
                last_sed = sed_np[-1]
                print(f"\n[DEBUG] Slot 0 SED values (last frame):")
                for i, (name, val) in enumerate(zip(CLASS_NAMES, last_sed)):
                    if val:
                        print(f"  {name}: {val}")
                
                # Show max SED across all frames
                max_sed = sed_np.max(axis=0)
                active_classes = np.where(max_sed)[0]
                if len(active_classes) > 0:
                    print(f"[DEBUG] Active classes (any frame): {[CLASS_NAMES[i] for i in active_classes]}")
                else:
                    print("[DEBUG] No active classes in any frame")
            
            # Get last frame (most recent)
            last_frame = -1
            for cls_idx in range(self.params['nb_classes']):
                if sed_np[last_frame, cls_idx]:
                    # Extract x, y for azimuth
                    x = doa_np[last_frame, cls_idx]
                    y = doa_np[last_frame, cls_idx + self.params['nb_classes']]
                    azimuth = np.degrees(np.arctan2(y, x))
                    detections.append({
                        'class': CLASS_NAMES[cls_idx],
                        'class_id': cls_idx,
                        'azimuth': azimuth,
                        'slot': slot
                    })
        
        return detections
    
    def display_results(self, detections):
        """Display detection results in terminal."""
        # Clear line and display
        sys.stdout.write('\r' + ' ' * 100 + '\r')
        
        if not detections:
            sys.stdout.write('[No sound detected]')
        else:
            parts = []
            for det in detections:
                direction = "←" if det['azimuth'] < -30 else "→" if det['azimuth'] > 30 else "↑"
                parts.append(f"{det['class']}:{det['azimuth']:+.0f}°{direction}")
            sys.stdout.write(' | '.join(parts))
        
        sys.stdout.flush()
    
    def process_loop(self):
        """Main processing loop."""
        print("\n[Processing started - Press Ctrl+C to stop]")
        
        while self.running:
            start_time = time.time()
            
            # Get audio from buffer
            with self.buffer_lock:
                audio = self.audio_buffer.copy()
            
            # Process
            try:
                features = self.extract_features(audio)
                result = self.infer(features)
                detections = self.parse_results(result)
                self.display_results(detections)
            except Exception as e:
                print(f"\nError: {e}")
            
            # Wait for next slide
            elapsed = time.time() - start_time
            sleep_time = (self.slide_interval_ms / 1000.0) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for microphone input."""
        import pyaudio
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        audio_data = audio_data.reshape(-1, 2).T  # (2, samples)
        
        # Update buffer (slide and append)
        with self.buffer_lock:
            self.audio_buffer = np.roll(self.audio_buffer, -frame_count, axis=1)
            self.audio_buffer[:, -frame_count:] = audio_data
        
        return (None, pyaudio.paContinue)
    
    def run_microphone(self, device_index=None):
        """Run real-time inference with microphone input."""
        try:
            import pyaudio
        except ImportError:
            print("Error: PyAudio not installed. Run: pip install pyaudio")
            return
        
        p = pyaudio.PyAudio()
        
        # Open stream
        chunk_size = int(self.sr * 0.1)  # 100ms chunks
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=self.sr,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size,
            stream_callback=self.audio_callback
        )
        
        self.running = True
        stream.start_stream()
        
        try:
            self.process_loop()
        except KeyboardInterrupt:
            print("\n\n[Stopped by user]")
        finally:
            self.running = False
            stream.stop_stream()
            stream.close()
            p.terminate()
    
    def run_file(self, audio_path):
        """Run inference on audio file."""
        print(f"\nProcessing file: {audio_path}")
        
        # Load audio
        audio, sr = utils.load_audio(audio_path, self.sr)
        if audio.shape[0] != 2:
            print(f"Error: Stereo audio required, got {audio.shape[0]} channels")
            return
        
        total_samples = audio.shape[1]
        slide_samples = int(self.sr * self.slide_interval_ms / 1000)
        
        print(f"Audio duration: {total_samples / self.sr:.1f}s")
        print("\n[Press Ctrl+C to stop]\n")
        
        self.running = True
        pos = 0
        
        try:
            while self.running and pos + self.window_samples <= total_samples:
                # Get window
                window = audio[:, pos:pos + self.window_samples]
                
                # Process
                features = self.extract_features(window)
                result = self.infer(features)
                detections = self.parse_results(result)
                
                # Display with timestamp
                timestamp = pos / self.sr
                sys.stdout.write(f'\r[{timestamp:6.1f}s] ')
                self.display_results(detections)
                
                pos += slide_samples
                time.sleep(self.slide_interval_ms / 1000.0)
                
        except KeyboardInterrupt:
            print("\n\n[Stopped by user]")
        
        print("\n[Done]")


def list_audio_devices():
    """List available audio input devices."""
    try:
        import pyaudio
    except ImportError:
        print("Error: PyAudio not installed. Run: pip install pyaudio")
        return
    
    p = pyaudio.PyAudio()
    print("\nAvailable audio input devices:")
    print("-" * 50)
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] >= 2:
            print(f"  [{i}] {info['name']}")
            print(f"      Channels: {info['maxInputChannels']}, Rate: {int(info['defaultSampleRate'])} Hz")
    
    p.terminate()


def main():
    parser = argparse.ArgumentParser(description='Real-time SELD inference')
    parser.add_argument('--file', type=str, help='Audio file to process (instead of microphone)')
    parser.add_argument('--device', type=int, default=None, help='Audio device index')
    parser.add_argument('--list-devices', action='store_true', help='List audio devices')
    parser.add_argument('--slide-ms', type=int, default=500, help='Slide interval in ms (default: 500)')
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoints/MSI_FAFS_DN_Final_audio_multiACCDOA_20251205_002553',
                        help='Checkpoint directory')
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
    
    # Initialize
    seld = RealtimeSELD(args.checkpoint, slide_interval_ms=args.slide_ms)
    
    if args.file:
        seld.run_file(args.file)
    else:
        seld.run_microphone(device_index=args.device)


if __name__ == '__main__':
    main()
