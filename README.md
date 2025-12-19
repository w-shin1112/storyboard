# 프로젝트명

📢 2025년 2학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다

🎉 2025년 2학기 AIKU - 열심히상 수상!

## 소개

영화, 드라마, 뮤직비디오 제작을 위한 AI 기반 스토리보드 스케치 자동 생성 시스템

Scene 정보(장면 설명, 대사)와 Shot 정보(Close-up shot, Medium shot, Full shot)를 입력받아 깨끗한 스케치 형태의 스토리보드 이미지를 자동으로 생성하는 파이프라인을 구축하였습니다. 

## 방법론
**[모델링 전략] Textual Embedding 기반의 정교한 샷(Shot) 제어**

- **Base Model & Fine-tuning Strategy**
    - **Backbone:** Stable Diffusion v1.5
    - **Optimization:** LoRA (Low-Rank Adaptation)를 적용하여 적은 연산량으로 목표 스토리보드 작화 스타일을 효율적으로 학습.

**Model Architecture**

<img width="1280" height="720" alt="pipeline" src="https://github.com/user-attachments/assets/54334050-e7e1-4f64-8c91-e3844c6a9b2e" />


- **핵심 방법론: Textual Embedding**

특히 본 프로젝트에서는 원하는 구도를 정확하게 생성해내기 위해 **Textual Embedding** 기법을 중점적으로 도입했습니다. 이 기법은 기존 DreamBooth 연구 등에서 제안된 '희귀 토큰(Rare Token)을 활용한 주체(Subject) 학습' 방식을 응용한 것입니다.

일반적인 텍스트 프롬프트만으로는 '클로즈업', '풀샷' 등의 카메라 워킹을 일관성 있게 제어하기 어렵습니다. 이를 해결하기 위해 우리는 **CLIP Text Encoder의 임베딩 레이어**에 샷(Shot) 정보를 담은 새로운 토큰을 추가하여 학습시켰습니다.

- **Custom Tokens:**
    - `<cu_trg>`: 얼굴 표정과 감정을 강조하는 **Close Up** 정보 학습
    - `<ms_trg>`: 인물의 동작과 상반신을 표현하는 **Medium Shot** 정보 학습
    - `<fs_trg>`: 인물의 전신과 공간감을 나타내는 **Full Shot** 정보 학습

이를 통해 모델은 사용자가 입력한 토큰에 맞춰 화풍(Style)은 유지하되, 스토리보드 연출에 필수적인 **구도(Composition) 정보**를 명확하게 분리하여 생성할 수 있게 되었습니다.

> ReferenceRuiz, N., et al. "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation." CVPR 2023.
>
## 환경 설정

프로젝트 실행을 위해 아래의 요구 사양 및 설치 단계를 확인해 주세요.

### 1. 요구 사항 (Prerequisites)

* **OS**: Linux (Ubuntu 권장)
* **GPU**: NVIDIA GPU (CUDA 11.8 호환, 최소 VRAM 24GB 권장)
* **Python**: v3.11
* **Conda**: Anaconda 또는 Miniconda 사용 권장

### 2. 설치 단계 (Installation)

**Step 1: Conda 가상환경 생성 및 활성화**

```bash
conda create -n storyboard python=3.11 -y
conda activate storyboard

```

**Step 2: PyTorch 및 CUDA 툴킷 설치**

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

```

**Step 3: 필수 패키지 설치**

```bash
pip install -r requirements.txt

```

**Step 4: Accelerate 환경 설정 (멀티 GPU 사용 시)**

```bash
accelerate config

```

---

## 사용 방법

### 1. 데이터 전처리 (Data Preprocessing)

학습 전, `Data_preprocessing_method/` 폴더 내의 스크립트를 사용하여 데이터셋을 준비합니다.

* **데이터 추출**: `python 00_extract_dataset.py`
* **태그 전처리**: `python 01_preprocess_tags.py`

### 2. 모델 학습 (Training)

제공된 `train.sh` 스크립트를 통해 학습을 시작합니다. `accelerate`를 사용하여 멀티 GPU 환경에서 최적화된 학습을 수행합니다.

```bash
# 스크립트 실행
bash train.sh

# 직접 실행 시 예시
python train.py \
  --pretrained_model_name_or_path "/path/to/model" \
  --train_data_dir "/path/to/Dataset" \
  --resolution 512 \
  --train_batch_size 6 \
  --num_train_epochs 25 \
  --mixed_precision "fp16" \
  --output_dir "./output"

```

### 3. 추론 (Inference)

학습된 체크포인트를 사용하여 이미지를 생성합니다.

```bash
# 스크립트 실행
bash inference.sh

# 직접 실행 시 예시
python inference.py \
  --base-model "/path/to/base_model" \
  --checkpoint "/path/to/checkpoint" \
  --trigger-word "<ms_trg>" \
  --prompt "medium shot, Eye level, a character standing in the forest" \
  --fuse-lora \
  --output "result.png"

```

### 4. 검증 및 배치 생성 (Validation)

* 배치 결과 생성: `python batch_generate.py`
* 검증 스크립트: `bash validation.sh`


## 예시 결과
**Input text = " Eye level, female, youth, happy, slim body, white shirt, black pants, no background, day time "**

<img width="1769" height="593" alt="image" src="https://github.com/user-attachments/assets/29988724-fc99-440b-b878-6677a67d3144" />

동일 Prompt에 대해 Shot 별로 Trigger Word를 설정하여 Inference한 결과물입니다.
Text를 잘 반영한 Consistent한 그림체로, Shot 별 차이가 확연히 드러나는 것을 확인할 수 있습니다. 

## 팀원

- [신명경] : Lead, Model Architecture, Data Augmentation
- [김태관] : Experiment, Pipeline Construction, Data Clustering
- [정성윤] : Image Preprocessing
- [박서연] : Text Data Preprocessing
- [장서현] : Data Preprocessing, Evaluation Metrics

