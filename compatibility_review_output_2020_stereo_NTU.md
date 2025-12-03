# output_2020_stereo와 NTU 모델 호환성 검토 보고서

## 요약

`output_2020_stereo` 데이터셋은 **NTU 모델과 기본적으로 호환 가능**합니다. 다만 디렉토리 구조와 설정 조정이 필요합니다. **Distance는 기본값(1)이지만, 모델이 distance를 처리하는 구조이므로 distance를 무시하려면 추가 설정이 필요합니다.**

---

## 1. 데이터 포맷 호환성

### ✅ 메타데이터 포맷: 호환 가능

**output_2020_stereo 메타데이터 구조:**
```
frame,class,source,azimuth,distance,onscreen
9,5,0,-62,1,0
10,5,0,-60,1,0
...
```

**NTU 모델이 기대하는 포맷:**
- CSV 헤더: `frame,class,source,azimuth,distance,onscreen` ✅ 일치
- 컬럼 순서: 동일 ✅
- 데이터 타입: 일치 ✅

**결론**: 메타데이터 포맷 완전 호환

---

### ⚠️ 디렉토리 구조: 부분 호환 (수정 필요)

**output_2020_stereo 구조:**
```
DCASE2020_Stereo_SELD_Dataset/
├── stereo_dev/
│   └── dev-train/          # ← 'dev-train' 폴더명
│       └── *.wav
├── metadata_dev/
│   └── dev-train/          # ← 'dev-train' 폴더명
│       └── *.csv
└── mel64_dnorm/            # 이미 추출된 feature들
    ├── stereo_dev/
    ├── stereo_dev_normalized/
    └── metadata_dev_adpit/
```

**NTU 모델이 기대하는 구조:**
```python
# extract_features.py에서 기대하는 경로:
audio_files = glob.glob(os.path.join(root_dir, 'stereo_dev', 'dev-*', '*.wav'))
label_files = glob.glob(os.path.join(root_dir, 'metadata_dev', 'dev-*', '*.csv'))

# data_generator.py에서 기대하는 경로:
audio_files = glob.glob(os.path.join(feat_dir, f'stereo_dev_normalized/{fold}*.pt'))
label_files = glob.glob(os.path.join(feat_dir, 'metadata_dev_adpit/{}*.pt'))
```

**분석**:
- `dev-*` 패턴은 `dev-train`을 포함하므로 ✅ 호환 가능
- Feature 파일들이 이미 추출되어 있음 ✅
- 다만 `parameters.py`의 `root_dir`와 `feat_dir` 경로를 올바르게 설정해야 함

---

### ✅ 오디오 포맷: 호환 가능

| 항목 | output_2020_stereo | NTU 모델 요구사항 | 호환성 |
|------|-------------------|------------------|--------|
| 채널 수 | 2 (Stereo) | 2 (Stereo) | ✅ |
| 샘플링 레이트 | 24kHz | 24kHz | ✅ |
| 파일 형식 | WAV | WAV | ✅ |
| 길이 | 5초 (dev) | 가변 | ✅ |

---

## 2. Distance 필드 처리

### ⚠️ Distance 사용 여부

**현재 상황:**
1. `output_2020_stereo`의 메타데이터: `distance=1` (기본값, 실제 거리 정보 없음)
2. NTU 모델은 distance를 처리하는 구조:
   - `model.py`: 출력에 distance 포함 (line 77-79)
   - `utils.py`: label 처리 시 distance 포함 (line 417, 426-430)
   - `loss.py`: loss 계산 시 distance 포함 가능

**Distance를 사용하지 않으려면:**

#### 현재 상황 분석
1. **Loss 함수 확인 결과:**
   - `SELDLossADPIT` 클래스에서 distance가 ACCDOA loss 계산에 포함됨 (line 93, 97)
   - Output: `(act*x, act*y, dist)` 3개 요소 모두 사용
   - Target: `(x, y, dist)` 포함

2. **모델 출력 구조:**
   - MultiACCDOA: `3 tracks × 3 (x, y, dist) × 13 classes = 117차원`
   - SingleACCDOA: `3 (x, y, dist) × 13 classes = 39차원`

#### 옵션 1: Distance 정규화만 비활성화 (간단, 권장)
```python
# parameters.py 또는 main.py 실행 시
params['dnorm'] = False  # Distance normalization 비활성화
```
- Distance 값(1)이 그대로 사용되지만, 모든 샘플이 동일하므로 학습에 큰 영향 없음
- Loss 계산에는 여전히 포함되지만, 실제로 학습이 진행되지 않음

#### 옵션 2: Loss 함수 수정 (Distance 완전 제외)
- `loss.py`의 `SELDLossADPIT.forward()` 메서드 수정 필요
- Line 93에서 `[:, :, :, 0:3, :]`를 `[:, :, :, 0:2, :]`로 변경 (dist 제외)
- Line 97에서 target의 distance 부분 제외

#### 옵션 3: Distance loss 가중치를 0으로 설정
- Loss 함수에 distance 가중치 파라미터 추가 필요
- 현재 코드 구조상 직접적인 가중치 파라미터는 없음

**실용적 권장사항:**
- ✅ **옵션 1 사용**: `dnorm=False` 설정만으로 충분
  - Distance=1이 모든 샘플에서 동일하므로, 모델이 이를 학습하더라도 실제 예측에는 영향 없음
  - 메트릭 계산 시 distance error는 이미 무시됨 (`lad_dist_thresh=inf`)

---

## 3. Feature 파일

### ✅ 이미 추출된 Feature 활용 가능

**output_2020_stereo에 이미 있는 Feature:**
- `mel64/stereo_dev/*.pt` - 원본 feature
- `mel64/stereo_dev_normalized/*.pt` - 정규화된 feature
- `mel64/metadata_dev_adpit/*.pt` - 처리된 label
- `mel64_dnorm/` - Distance normalization 적용된 버전

**NTU 모델 사용 시:**
- Feature 추출 스킵하고 기존 `.pt` 파일 직접 사용 가능
- `parameters.py`에서 `feat_dir`를 `output_2020_stereo/DCASE2020_Stereo_SELD_Dataset/mel64_dnorm`로 설정

---

## 4. 필수 수정 사항

### 1) parameters.py 설정 변경

```python
params = {
    'root_dir': 'dcase2025_stereo_seld_data_generator/output_2020_stereo/DCASE2020_Stereo_SELD_Dataset',
    'feat_dir': 'dcase2025_stereo_seld_data_generator/output_2020_stereo/DCASE2020_Stereo_SELD_Dataset/mel64_dnorm',
    # 또는 distance를 사용하지 않으려면:
    # 'feat_dir': 'dcase2025_stereo_seld_data_generator/output_2020_stereo/DCASE2020_Stereo_SELD_Dataset/mel64',
    
    # Distance normalization 비활성화 (distance 사용 안 할 경우)
    'dnorm': False,  # 또는 True로 유지해도 무방 (distance=1이 정규화됨)
    
    # 다른 설정들...
    'nb_classes': 13,  # DCASE2020은 13개 클래스
    'label_sequence_length': 50,  # 5초 = 50 frames @ 100ms
}
```

### 2) Feature 재추출 vs 기존 Feature 사용

**옵션 A: 기존 Feature 사용 (권장)**
- 이미 추출된 `.pt` 파일이 있으므로 feature 추출 단계 스킵
- `extract_features.py`를 실행하지 않거나, 파일이 이미 있으면 스킵됨

**옵션 B: Feature 재추출**
- NTU 모델의 feature extraction 파라미터와 일치시키려면 재추출 필요
- `extract_features.py`에서 자동으로 기존 파일 스킵하므로 안전

### 3) Fold 구조 확인

NTU 모델은 fold 기반으로 데이터를 분할:
```python
'dev_train_folds': ['fold3', 'fold5', 'fold6'],
'dev_test_folds': ['fold4'],
```

`output_2020_stereo`의 파일명 패턴:
- `fold1_room1_mix001_ov1_deg215_start0056.wav`

따라서 fold 정보가 파일명에 포함되어 있어 ✅ 호환 가능

---

## 5. 호환성 체크리스트

| 항목 | 상태 | 비고 |
|------|------|------|
| 메타데이터 포맷 | ✅ | 완전 호환 |
| 오디오 포맷 | ✅ | 완전 호환 |
| 디렉토리 구조 | ⚠️ | 경로 설정 필요 |
| Feature 파일 | ✅ | 이미 추출됨 |
| Distance 필드 | ⚠️ | 기본값 1, 처리 구조는 있음 |
| Fold 구조 | ✅ | 파일명에 포함 |
| 클래스 수 | ✅ | 13개 일치 |
| Label 길이 | ✅ | 50 frames 일치 |

---

## 6. 사용 권장사항

### Distance를 사용하지 않는 경우

1. **Distance Normalization 비활성화:**
   ```python
   params['dnorm'] = False
   ```

2. **Feature 디렉토리 선택:**
   - `mel64` 사용 (distance normalization 없음)
   - 또는 `mel64_dnorm` 사용해도 무방 (distance=1이 정규화되지만 영향 적음)

3. **Loss 함수 확인:**
   - Distance loss가 포함되어 있다면, 가중치를 0으로 설정하거나 loss 함수 수정 필요
   - 현재 코드에서는 distance가 loss에 포함될 가능성이 높음

### 실행 순서

1. `parameters.py`에서 경로 설정
2. 기존 feature 파일 확인 (이미 있으면 재추출 불필요)
3. Distance 관련 설정 조정
4. 학습 시작

---

## 7. 잠재적 이슈 및 해결방법

### 이슈 1: Distance 값이 학습에 영향

**문제**: Distance=1이 모든 샘플에서 동일하여 학습에 도움이 되지 않음

**해결**:
- Loss 함수에서 distance loss 가중치를 0으로 설정
- 또는 모델 출력에서 distance 부분을 무시

### 이슈 2: 디렉토리 경로 불일치

**문제**: 상대 경로 설정이 잘못될 수 있음

**해결**:
- 절대 경로 사용 또는 상대 경로를 현재 작업 디렉토리 기준으로 정확히 설정

### 이슈 3: Feature 파라미터 불일치

**문제**: 기존 feature와 NTU 모델의 feature 파라미터가 다를 수 있음

**해결**:
- Feature 재추출 권장 (기존 파일 자동 스킵되므로 안전)
- 또는 feature 파라미터 확인 후 필요시 재추출

---

## 결론

**`output_2020_stereo`는 NTU 모델과 호환 가능합니다.** 

**주요 사항:**
1. ✅ 데이터 포맷은 완전 호환
2. ⚠️ 경로 설정 필요
3. ⚠️ Distance 필드는 기본값 1이지만, 모델이 처리하므로 설정 조정 필요
4. ✅ Feature 파일이 이미 추출되어 있어 바로 사용 가능

**Distance를 사용하지 않으려면:**
- `dnorm=False` 설정
- Loss 함수에서 distance 부분 제외 또는 가중치 0 설정
- 메트릭에서 distance error 무시 (이미 `lad_dist_thresh=inf`로 설정됨)

