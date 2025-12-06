"""
benchmark_inference.py

Benchmark script to measure inference time for real-time SELD.
Tests feature extraction and model inference separately.

Usage:
    python benchmark_inference.py
"""

import os
import time
import torch
import numpy as np
import joblib
import glob
from parameters import params
from model import SELDModel
import utils


def benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # ============ 1. Load Model ============
    print("\n[1] Loading model...")
    checkpoint_dir = os.path.join(params['checkpoints_dir'], 
                                   'MSI_FAFS_DN_Final_audio_multiACCDOA_20251205_002553')
    model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    # Load saved config from checkpoint
    config_path = os.path.join(checkpoint_dir, 'config.pkl')
    import pickle
    with open(config_path, 'rb') as f:
        saved_params = pickle.load(f)
    print(f"Loaded config: {config_path}")
    print(f"  Features: gamma={saved_params.get('gamma')}, ipd={saved_params.get('ipd')}, "
          f"iv={saved_params.get('iv')}, slite={saved_params.get('slite')}, ms={saved_params.get('ms')}")
    
    # Calculate input channels based on saved params
    n_channels = 2  # stereo mel
    if saved_params.get('ms'): n_channels += 2  # mid-side
    if saved_params.get('gamma'): n_channels += 1
    if saved_params.get('ipd'): n_channels += 2
    if saved_params.get('iv'): n_channels += 1
    if saved_params.get('slite'): n_channels += 1
    print(f"  Input channels: {n_channels}")
    
    # Model input shape: (batch, channels, time, mels)
    in_feat_shape = (1, n_channels, 251, saved_params['nb_mels'])
    model = SELDModel(saved_params, in_feat_shape)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['seld_model'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from: {model_path}")
    
    # ============ 2. Load Scaler ============
    print("\n[2] Loading scaler...")
    # Try saved_params feat_dir first, then fall back to current params
    scaler_path = os.path.join(saved_params['feat_dir'], 'scaler_dev.pkl')
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(params['feat_dir'], 'scaler_dev.pkl')
    if not os.path.exists(scaler_path):
        # Search for scaler in parent directory
        parent_dir = os.path.dirname(saved_params['feat_dir'])
        for folder in os.listdir(parent_dir):
            candidate = os.path.join(parent_dir, folder, 'scaler_dev.pkl')
            if os.path.exists(candidate):
                scaler_path = candidate
                break
    scalers = joblib.load(scaler_path)
    print(f"Scaler loaded from: {scaler_path}")
    
    # ============ 3. Find Test Audio ============
    print("\n[3] Finding test audio file...")
    audio_dir = os.path.join(saved_params['root_dir'], 'stereo_dev')
    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return
    test_audio = audio_files[0]
    print(f"Test file: {os.path.basename(test_audio)}")
    
    # ============ 4. Benchmark Feature Extraction ============
    print("\n[4] Benchmarking feature extraction...")
    
    # Audio params
    hop_length = int(saved_params['sampling_rate'] * saved_params['hop_length_s'])
    n_fft = 2 ** (2 * hop_length - 1).bit_length()
    
    # Warmup
    audio, sr = utils.load_audio(test_audio, saved_params['sampling_rate'])
    _ = utils.extract_stereo_features(
        audio, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        nb_mels=saved_params['nb_mels'], max_freq=saved_params['max_freq'],
        use_gamma=saved_params['gamma'], use_ipd=saved_params['ipd'],
        use_iv=saved_params['iv'], use_slite=saved_params['slite'], use_ms=saved_params['ms']
    )
    
    # Benchmark
    n_runs = 10
    feat_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        audio, sr = utils.load_audio(test_audio, saved_params['sampling_rate'])
        features = utils.extract_stereo_features(
            audio, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
            nb_mels=saved_params['nb_mels'], max_freq=saved_params['max_freq'],
            use_gamma=saved_params['gamma'], use_ipd=saved_params['ipd'],
            use_iv=saved_params['iv'], use_slite=saved_params['slite'], use_ms=saved_params['ms']
        )
        feat_times.append(time.perf_counter() - start)
    
    feat_time_avg = np.mean(feat_times) * 1000  # ms
    feat_time_std = np.std(feat_times) * 1000
    print(f"Feature extraction: {feat_time_avg:.1f} ± {feat_time_std:.1f} ms")
    print(f"Feature shape: {features.shape}")
    
    # ============ 5. Benchmark Normalization ============
    print("\n[5] Benchmarking normalization...")
    norm_times = []
    for _ in range(n_runs):
        feat_copy = features.copy()
        start = time.perf_counter()
        for ch, scaler in enumerate(scalers):
            channel_data = feat_copy[ch].reshape(-1, feat_copy.shape[2])
            feat_copy[ch] = scaler.transform(channel_data).reshape(feat_copy.shape[1], feat_copy.shape[2])
        norm_times.append(time.perf_counter() - start)
    
    norm_time_avg = np.mean(norm_times) * 1000
    norm_time_std = np.std(norm_times) * 1000
    print(f"Normalization: {norm_time_avg:.1f} ± {norm_time_std:.1f} ms")
    
    # ============ 6. Benchmark Model Inference ============
    print("\n[6] Benchmarking model inference...")
    
    # Prepare normalized tensor
    feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Warmup
    with torch.no_grad():
        _ = model(feat_tensor)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    infer_times = []
    for _ in range(n_runs):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = model(feat_tensor)
        if device == 'cuda':
            torch.cuda.synchronize()
        infer_times.append(time.perf_counter() - start)
    
    infer_time_avg = np.mean(infer_times) * 1000
    infer_time_std = np.std(infer_times) * 1000
    print(f"Model inference: {infer_time_avg:.1f} ± {infer_time_std:.1f} ms")
    print(f"Output shape: {output.shape}")
    
    # ============ 7. Benchmark Output Parsing ============
    print("\n[7] Benchmarking output parsing...")
    parse_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = utils.get_multiaccdoa_labels(output, saved_params['nb_classes'], dnorm=saved_params['dnorm'], params=saved_params)
        parse_times.append(time.perf_counter() - start)
    
    parse_time_avg = np.mean(parse_times) * 1000
    parse_time_std = np.std(parse_times) * 1000
    print(f"Output parsing: {parse_time_avg:.1f} ± {parse_time_std:.1f} ms")
    
    # ============ Summary ============
    total_time = feat_time_avg + norm_time_avg + infer_time_avg + parse_time_avg
    print("\n" + "="*50)
    print("SUMMARY (5-second audio window)")
    print("="*50)
    print(f"  Feature extraction:  {feat_time_avg:7.1f} ms")
    print(f"  Normalization:       {norm_time_avg:7.1f} ms")
    print(f"  Model inference:     {infer_time_avg:7.1f} ms")
    print(f"  Output parsing:      {parse_time_avg:7.1f} ms")
    print("-"*50)
    print(f"  TOTAL:               {total_time:7.1f} ms")
    print("="*50)
    
    if total_time < 1000:
        print(f"✅ 실시간 가능! (5초 윈도우 처리에 {total_time:.0f}ms)")
        max_fps = 5000 / total_time
        print(f"   최대 {max_fps:.1f}x 실시간 처리 가능")
    else:
        print(f"⚠️  처리 시간이 1초 초과 ({total_time:.0f}ms)")


if __name__ == '__main__':
    benchmark()
