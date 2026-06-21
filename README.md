# PMSM 영구자석 온도 예측

전류·전압 시계열을 이용해 영구자석 동기 모터(Permanent Magnet Synchronous Motor, PMSM)의 영구자석 온도(`pm`)를 예측하는 PyTorch 기반 연구 프로젝트입니다.

이 프로젝트는 센서 데이터 전처리, PM 온도 패턴 기반 실험 군집화, LSTM Seq2Seq 모델 학습으로 구성됩니다. 기본 Seq2Seq 모델에서 시작하여 Scheduled Teacher Forcing과 Additive Attention을 순차적으로 적용합니다.

## 프로젝트 목적

모터 내부의 영구자석 온도는 직접 측정하기 어렵지만, 과도한 온도 상승은 자석의 성능 저하와 모터 수명 감소로 이어질 수 있습니다. 본 프로젝트에서는 모터 운전 중 측정되는 d-q축 전류와 전압을 이용해 미래의 영구자석 온도 변화를 예측합니다.

```text
과거 u_q, u_d, i_q, i_d 시계열
                ↓
          Seq2Seq 모델
                ↓
         미래 PM 온도 시계열
```

## 전체 파이프라인

```text
Raw 데이터(2 Hz)
      ↓
4배 Downsampling(0.5 Hz, 2초 간격)
      ↓
Robust 정규화
      ↓
Soft 재정규화
      ↓
feature별 Robust/Soft 선택
      ↓
PM 시계열 FastDTW 군집화
      ↓
군집 비율 기반 Train/Validation/Test ID 구성
      ↓
Seq2Seq 모델 학습 및 비교
```

## 저장소 구성

```text
.
├── data/
│   ├── downsampled.py     # 4배 다운샘플링
│   ├── robust.py          # profile_id·feature별 Robust 정규화
│   ├── soft.py            # profile_id별 Soft 재정규화
│   ├── std_ratio.py       # feature와 PM의 표준편차 비율 분석
│   ├── scale_score.py     # 선택한 ID의 스케일 진단
│   └── Data-kaggle-url.docx
├── cluster/
│   ├── code/
│   │   ├── cluster_mix_final.py # 혼합 정규화 및 PM 군집화
│   │   └── subset_maker.py      # 군집 비율 기반 데이터 분할
│   └── image/                    # 덴드로그램과 군집별 시계열
├── model/
│   ├── model_1.py         # 기본 LSTM Seq2Seq
│   ├── model_2.py         # Scheduled Teacher Forcing
│   └── model_3.py         # 2단 Additive Attention Seq2Seq
└── README.md
```

## 데이터

하나의 `profile_id`는 한 번의 모터 실험을 의미하며, 각 행은 실험 시작부터 종료까지 시간순으로 측정된 센서값입니다.

주요 변수는 다음과 같습니다.

| 구분 | 변수 | 설명 |
|---|---|---|
| 입력 | `i_d`, `i_q` | d-q축 전류 |
| 입력 | `u_d`, `u_q` | d-q축 전압 |
| 목표 | `pm` | 영구자석 온도 |
| 식별자 | `profile_id` | 실험 ID |

원본 데이터는 2 Hz로 측정되었습니다. `downsampled.py`는 4개마다 하나의 샘플을 선택하므로 전처리 후 샘플 간격은 2초, 샘플링 주파수는 0.5 Hz입니다.

원본 데이터는 저장소에 포함하지 않았으며, 데이터 출처는 `data/Data-kaggle-url.docx`에서 확인할 수 있습니다.

## 전처리

### 1. Robust 정규화

전류, 전압, 온도는 단위와 값의 범위가 서로 다릅니다. 또한 센서 데이터에는 노이즈와 이상치가 포함될 수 있으므로, 평균과 표준편차 대신 중앙값과 IQR을 사용하는 Robust 정규화를 적용합니다.

```math
x_{robust}=\frac{x-\operatorname{median}(x)}{Q_3(x)-Q_1(x)}
```

정규화는 `profile_id-feature` 단위로 수행합니다.

### 2. Soft 재정규화

d-q축 전류와 전압은 특정 운전 구간에서 거의 일정한 값을 가질 수 있습니다. 이 경우 IQR이 0에 가까워지면서 Robust 정규화 결과가 과도하게 증폭될 수 있습니다.

이를 완화하기 위해 한 `profile_id`의 Robust 정규화된 feature를 결합하여 표준편차 `soft_std`를 구하고, 동일 ID의 feature를 이 값으로 다시 나눕니다.

```math
x_{soft}=\frac{x_{robust}}{soft\_std}
```

### 3. 혼합 정규화 데이터셋

Soft 재정규화를 모든 feature에 일괄 적용하면 이미 정상적인 크기를 가진 feature까지 지나치게 축소될 수 있습니다. 따라서 Robust 정규화 후 표준편차가 4 이상인 feature에는 Soft 값을 사용하고, 나머지 feature에는 Robust 값을 유지합니다.

```math
x_{final}=\begin{cases}
x_{soft}, & \operatorname{Std}(x_{robust})\ge4 \\
x_{robust}, & \operatorname{Std}(x_{robust})<4
\end{cases}
```

임계값 4는 전체 실험의 정규화 후 표준편차 분포와 시계열을 분석하여 경험적으로 선정했습니다. 각 `profile_id-feature`에서 사용할 정규화 방식은 `normalization_verification.xlsx`로 관리합니다.

## PM 시계열 군집화

실험마다 PM 초기 온도, 상승 속도, 최고 온도와 변화 형태가 다릅니다. ID를 단순 무작위로 분할하면 특정 PM 패턴이 Train 또는 Test 데이터에 편중될 수 있습니다.

이를 줄이기 위해 다음 과정으로 실험 ID를 군집화합니다.

1. 혼합 정규화 데이터에서 ID별 PM 시계열을 추출합니다.
2. 모든 ID 조합의 FastDTW 거리를 계산합니다.
3. 거리 행렬에 계층적 군집화를 적용합니다.
4. 전체 ID를 8개 PM 패턴 군집으로 분류합니다.
5. 각 군집의 크기에 비례해 ID를 선택하여 학습·검증·시험 데이터를 구성합니다.

FastDTW는 온도 변화 시점이나 실험 길이가 조금 다른 경우에도 시간축을 유연하게 정렬하여 시계열 형태를 비교합니다. 데이터 분할은 행이 아닌 `profile_id` 단위로 수행하여 동일 실험이 Train과 Test에 동시에 포함되는 것을 방지합니다.

군집화 결과는 `cluster/image/`에서 확인할 수 있습니다.

## 예측 문제 정의

세 모델의 공통 입력과 출력은 다음과 같습니다.

| 항목 | 설정 |
|---|---|
| 입력 feature | `u_q`, `u_d`, `i_q`, `i_d` |
| 예측 target | `pm` |
| 입력 길이 | 100 samples, 약 200초 |
| 예측 길이 | 50 samples, 약 100초 |
| Hidden dimension | 64 |
| LSTM layers | 2 |
| Batch size | 32 |
| Epochs | 50 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | MSE |
| 보조 지표 | MAE |

각 ID에는 한 칸 간격의 sliding window를 적용합니다.

```text
입력 0~99   → PM 100~149 예측
입력 1~100  → PM 101~150 예측
입력 2~101  → PM 102~151 예측
```

## 모델

### Model 1 — 기본 LSTM Seq2Seq

`model/model_1.py`

- 2-layer LSTM Encoder–Decoder
- Encoder의 마지막 hidden/cell state를 Decoder로 전달
- 학습 시 정답 PM을 다음 Decoder 입력으로 사용하는 Teacher Forcing 적용
- 기본 Seq2Seq 성능을 확인하기 위한 기준 모델

```text
전류·전압 100개
      ↓
2-layer LSTM Encoder
      ↓
2-layer LSTM Decoder
      ↓
미래 PM 50개
```

### Model 2 — Scheduled Teacher Forcing

`model/model_2.py`

Model 1과 동일한 LSTM Seq2Seq 구조를 사용하지만, 학습이 진행될수록 Teacher Forcing 비율을 감소시킵니다.

```text
학습 초기: 실제 PM을 다음 입력으로 주로 사용
학습 후기: 모델의 직전 예측값을 다음 입력으로 주로 사용
검증 단계: 모델의 예측값만 사용해 autoregressive 예측
```

학습과 실제 추론 조건의 차이에서 발생하는 exposure bias를 줄이고, 여러 시점에 걸친 연속 예측에 적응하도록 구성했습니다.

### Model 3 — 2단 Additive Attention Seq2Seq

`model/model_3.py`

Model 2의 Scheduled Teacher Forcing에 Bahdanau Additive Attention을 추가한 모델입니다.

- Encoder는 과거 100개 시점의 전체 출력을 반환합니다.
- Decoder의 두 LSTM 계층이 각각 별도의 Attention을 사용합니다.
- 각 미래 PM을 예측할 때 중요한 과거 전류·전압 구간을 선택적으로 참조합니다.

```text
Encoder 전체 출력
    ├─ Attention 1 → LSTMCell 1
    └─ Attention 2 → LSTMCell 2
                         ↓
                     PM 예측
```

## 모델 비교

| 구분 | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| 기본 구조 | LSTM Seq2Seq | LSTM Seq2Seq | Attention Seq2Seq |
| Teacher Forcing | 고정 | 점진적 감소 | 점진적 감소 |
| Attention | 없음 | 없음 | 2단 Additive Attention |
| 자율 연속 예측 | 제한적 | 지원 | 지원 |
| 과거 전체 시점 활용 | 마지막 state에 압축 | 마지막 state에 압축 | Attention으로 직접 참조 |

## 설치

Python 3.9 이상 환경을 권장합니다.

```bash
pip install torch pandas numpy matplotlib scipy fastdtw scikit-learn openpyxl
```

CUDA를 지원하는 PyTorch 환경에서는 GPU를 자동으로 사용하고, 그렇지 않으면 CPU를 사용합니다.

## 실행

현재 스크립트의 데이터 및 결과 경로는 절대 경로로 작성되어 있습니다. 실행 전에 각 파일의 `base_path`, `input_path`, `output_path`를 로컬 환경에 맞게 변경해야 합니다.

### 전처리

```bash
python data/downsampled.py
python data/robust.py
python data/soft.py
```

### 군집화 및 데이터 분할

```bash
python cluster/code/cluster_mix_final.py
python cluster/code/subset_maker.py
```

### 모델 학습

```bash
python model/model_1.py
python model/model_2.py
python model/model_3.py
```

각 모델은 학습 후 다음 파일을 저장합니다.

- `model.pt`: 학습된 모델 가중치
- `loss_graph.png`: Train/Validation loss 및 MAE 그래프
- `log.txt`: epoch별 학습 기록
- `info.txt`: Train/Validation/Test ID 목록

## 관련 논문

본 프로젝트와 관련된 Adaptive Seq2Seq 장기 예측 연구는 아래 링크에서 확인할 수 있습니다.

- [Adaptive sequence-to-sequence learning for long-term time series prediction](https://pubs.aip.org/aip/adv/article/16/4/045101/3385855/Adaptive-sequence-to-sequence-learning-for-long)

## 참고 사항

- 연구 및 실험 목적으로 작성된 코드입니다.
- 원본 데이터와 전처리 결과 파일은 저장소에 포함하지 않았습니다.
- 재현을 위해 데이터 경로와 `normalization_verification.xlsx`를 준비해야 합니다.
- Test ID는 학습 코드에 정의되어 있으나, 별도의 테스트 평가 및 역정규화 코드는 추가로 구현할 수 있습니다.

