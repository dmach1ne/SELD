import os
import glob
import re
from collections import defaultdict

# 파일명에서 start 시간 추출하여 패턴 분석
feat_dir = '../dcase2025_stereo_seld_data_generator/output_2020_stereo/DCASE2020_Stereo_SELD_Dataset/mel64'

# 실제 파일이 있는지 확인
stereo_dir = os.path.join(feat_dir, 'stereo_dev_normalized')
if not os.path.exists(stereo_dir):
    print(f"Directory not found: {stereo_dir}")
    print("Trying alternative paths...")
    # 다른 경로 시도
    alt_paths = [
        'dcase2025_stereo_seld_data_generator/output_2020_stereo/DCASE2020_Stereo_SELD_Dataset/mel64/stereo_dev_normalized',
        '../dcase2025_stereo_seld_data_generator/output_2020_stereo/DCASE2020_Stereo_SELD_Dataset/mel64/stereo_dev_normalized',
    ]
    for alt_path in alt_paths:
        if os.path.exists(alt_path):
            stereo_dir = alt_path
            feat_dir = os.path.dirname(alt_path)
            break

if os.path.exists(stereo_dir):
    print(f"Found directory: {stereo_dir}\n")
    
    # 파일명에서 start 시간 추출
    files = glob.glob(os.path.join(stereo_dir, 'fold1*.pt'))
    
    if files:
        print(f"Found {len(files)} files. Analyzing start time patterns...\n")
        
        # start 시간 추출
        start_times = []
        base_names = defaultdict(list)  # 같은 base name을 가진 파일들 그룹화
        
        for f in files[:100]:  # 처음 100개만 분석
            filename = os.path.basename(f)
            # startXXXX 패턴 찾기
            match = re.search(r'start(\d+)', filename)
            if match:
                start_time = int(match.group(1))
                start_times.append(start_time)
                
                # base name 추출 (start 부분 제외)
                base_name = re.sub(r'_start\d+', '', filename)
                base_names[base_name].append(start_time)
        
        if start_times:
            print("="*60)
            print("Start Time Analysis")
            print("="*60)
            print(f"Total files analyzed: {len(start_times)}")
            print(f"Min start time: {min(start_times)}")
            print(f"Max start time: {max(start_times)}")
            print(f"Start time range: {max(start_times) - min(start_times)}")
            print(f"\nStart time distribution (first 20):")
            for st in sorted(start_times)[:20]:
                print(f"  {st}")
            
            print(f"\n\nBase file groups (same audio, different start times):")
            print("="*60)
            for base_name, starts in list(base_names.items())[:5]:
                if len(starts) > 1:
                    print(f"\nBase: {base_name[:60]}...")
                    print(f"  Start times: {sorted(starts)}")
                    print(f"  Count: {len(starts)}")
                    print(f"  Range: {min(starts)} - {max(starts)}")
                    print(f"  Interval: {sorted(starts)[1] - sorted(starts)[0] if len(starts) > 1 else 'N/A'}")
        else:
            print("No 'start' pattern found in filenames")
    else:
        print(f"No files found in {stereo_dir}")
else:
    print(f"Directory not found: {stereo_dir}")
    print("\nPlease check the path to the data generator output directory.")

