# BoostTrack++: 실시간 객체 추적 및 속도 분석
[BoostTrack 공식 저장소](https://github.com/vukasin-stanojevic/BoostTrack)

BoostTrack++ 다중 객체 추적(MOT) 기술을 활용하여 실시간으로 비디오에서 객체를 감지하고 추적하는 시스템입니다. ROI(관심 영역)를 설정하여 해당 영역 내에서 이동하는 객체의 속도를 계산하고, 결과를 시각적으로 표시합니다.

## 환경 설정
```bash
# 저장소 클론
$ git clone https://github.com/vukasin-stanojevic/BoostTrack.git
$ cd BoostTrack

# Conda 가상환경 생성 및 패키지 설치
$ conda env create -f boost-track-env.yml
$ conda activate boosttrack

# Streamlit 추가 설치
$ pip install streamlit
```

<br>

## 모델 가중치 다운로드
BoostTrack++을 실행하려면 모델 가중치 파일을 다운로드해야 합니다.

모델 가중치는 아래 링크에서 다운로드할 수 있습니다:
**[BoostTrack 모델 가중치 다운로드](https://drive.google.com/drive/folders/15hZcR4bW_Z9hEaXXjeWhQl_jwRKllauG)**

#### 다운로드 후 저장할 경로
```bash
mkdir -p external/weights
```

| 모델 종류 | 파일명 | 저장 경로 |
|-----------|--------------------------|----------------------------|
| **ReID 모델** | `mot20_sbs_S50.pth` | `./external/weights/mot20_sbs_S50.pth` |
| **ByteTrack MOT 모델** | `bytetrack_x_mot20.tar` | `./external/weights/bytetrack_x_mot20.tar` |

<br>

## BoostTrack++ 실행 방법
BoostTrack++에서는 `app.py`를 실행하여 사용자 인터페이스를 제공합니다.

```bash
$ streamlit run app.py
```

실행 후, Streamlit 웹 애플리케이션이 브라우저에서 열립니다. 사용자는 비디오 파일을 업로드하여 실시간으로 객체를 추적하고 속도를 분석할 수 있습니다.

<br>

## 주요 기능 및 파일 설명
#### `app.py`: Streamlit 기반 GUI 인터페이스로 비디오 파일을 실시간으로 추적 및 속도 측정
1. 사용자는 `.mp4`, `.avi`, `.mov` 등의 비디오 파일을 업로드
2. 비디오 내 객체를 감지하고, ROI 내부의 객체만 선택하여 추적
3. **속도 계산**: 3초마다 ROI 내부의 객체의 이동 거리를 기반으로 속도를 계산
4. **결과 저장**:
   - 추적된 비디오는 `data/output` 폴더에 저장
   - 속도 분석 결과는 `data/output/speed_log.txt`로 저장

#### `check_roi.py`: ROI(관심 영역) 설정을 위한 GUI 툴 (사용자가 직접 ROI 좌표 설정 가능)
- `data/videos` 폴더에 있는 비디오의 첫 번째 프레임을 로드하여 사용자가 관심 영역(ROI)을 설정
- 마우스로 4개의 점을 선택하여 ROI를 설정한 후, `n` 키를 누르면 ROI 값이 로그로 출력됨

<br>

## 결과 저장 및 로그
BoostTrack++은 추적 결과는 다음 경로에 저장됩니다.
- **비디오 파일**: `data/output/output_video.mp4`
- **속도 로그 파일**: `data/output/speed_log.txt`

#### 속도 로그 예시
```txt
🔹 3초 시점 속도 로그:
객체 ID 35: 이동 거리 44.01 px, 속도 14.67 px/s
객체 ID 31: 이동 거리 58.41 px, 속도 19.47 px/s
평균 속도: 16.68 px/s
...
===== 최종 결과 =====
최종 평균 속도: 13.54 px/s
ROI 직선 길이: 442.12 px
예상 대기 시간: 32.66 초
```