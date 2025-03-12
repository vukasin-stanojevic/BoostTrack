import streamlit as st
import cv2
import torch
import tempfile
import os
import numpy as np
import time
from external.adaptors.detector import Detector
from tracker.boost_track import BoostTrack

# ROI 다각형 좌표 설정 (예시)
ROI_POLYGON = np.array([(613, 19), (489, 37), (572, 329), (718, 296)], np.int32)

# 속도 계산 관련 파라미터
FPS = 30                   # 비디오 FPS (기본값, 실제 FPS는 비디오에서 가져옴)
UPDATE_INTERVAL = 3 * FPS  # 3초마다 속도 갱신 (프레임 단위)
MIN_DISTANCE = 5.0         # 최소 이동거리 (5px 미만은 0으로 처리)

def is_inside_roi(point, roi_polygon):
    """
    주어진 점이 ROI(관심 영역) 다각형 내부에 포함되는지 여부를 확인

    Args:
        point (tuple): 검사할 점의 (x, y) 좌표
        roi_polygon (np.ndarray): ROI를 정의하는 다각형 좌표 배열

    Returns:
        bool: 점이 ROI 내부에 있으면 True, 그렇지 않으면 False
    """
    return cv2.pointPolygonTest(roi_polygon, (int(point[0]), int(point[1])), False) >= 0

def calculate_roi_mid_length(roi_polygon):
    """
    ROI 다각형의 왼쪽과 오른쪽 경계선의 중간점을 찾아, 두 점 사이의 거리를 계산
    이 거리는 ROI 내부 직선으로 대기열 길이를 측정하는 데 사용됨

    Args:
        roi_polygon (np.ndarray): ROI를 정의하는 다각형 좌표 배열

    Returns:
        tuple: (ROI 내부 직선 길이, 왼쪽 중간점 좌표, 오른쪽 중간점 좌표)
    """
    left_mid = ((roi_polygon[0][0] + roi_polygon[1][0]) // 2,
                (roi_polygon[0][1] + roi_polygon[1][1]) // 2)
    right_mid = ((roi_polygon[2][0] + roi_polygon[3][0]) // 2,
                 (roi_polygon[2][1] + roi_polygon[3][1]) // 2)
    roi_length = np.sqrt((right_mid[0]-left_mid[0])**2 + (right_mid[1]-left_mid[1])**2)
    return roi_length, left_mid, right_mid


st.title("BoostTrack++ ROI 기반 실시간 객체 추적 및 속도 측정")
st.write("비디오를 업로드하고 '모델 실행' 버튼을 클릭하면 BoostTrack++이 ROI 영역 내에서 객체를 추적하고, 각 객체 및 평균 속도를 계산하여 표시합니다.")

uploaded_file = st.file_uploader("비디오 파일을 업로드하세요", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)  # 원본 비디오 미리보기
    if st.button("모델 실행"):
        # 모델 및 추적기 초기화
        detector = Detector("yolox", "external/weights/bytetrack_x_mot20.tar", "custom")
        tracker = BoostTrack()

        # 업로드된 비디오를 임시 파일로 저장
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

        # 비디오 파일 로드 및 버퍼 설정
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 결과 비디오 저장 설정 -> 임시 폴더에 저장 후, 나중에 data/output으로 이동
        output_video_path = os.path.join(tempfile.gettempdir(), "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

        # 실시간 영상 표시를 위한 Streamlit placeholder 생성
        frame_display = st.empty()

        # 속도 계산 관련 변수 초기화
        frame_count = 0
        last_update_frame = 0
        object_positions = {}   # {object_id: {"start": (x, y), "last": (x, y)}}
        object_speeds = {}      # {object_id: speed (px/s)}
        object_last_frame = {}  # {object_id: 마지막 업데이트된 frame 번호}
        avg_speed_text = "Avg Speed: 0.00 px/s"
        speed_log = ""          # 속도 로그를 누적해서 저장할 문자열

        # ROI 직선 길이 및 중간 선 좌표 계산 -> 대기열 길이
        roi_length, left_mid, right_mid = calculate_roi_mid_length(ROI_POLYGON)

        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # ROI 영역 그리기 (ROI 다각형과 ROI 중간 선(파란색) 표시)
            cv2.polylines(frame, [ROI_POLYGON], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.line(frame, left_mid, right_mid, (255, 0, 0), 2)

            # 프레임 해상도 조정 및 색상 변환 (BGR → RGB)
            img_numpy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # YOLOX 입력용 텐서 변환 -> GPU 사용
            img = torch.from_numpy(img_numpy).permute(2, 0, 1).float().cuda()
            img /= 255.0
            if len(img.shape) == 3:
                img = img.unsqueeze(0)

            # 객체 탐지를 위한 YOLOX 추론
            with torch.no_grad():
                outputs = detector.detect(img)
            if outputs is None:
                frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                continue

            # BoostTrack++ 추적 실행
            targets = tracker.update(outputs, img, img_numpy, "custom_video")
            if isinstance(targets, np.ndarray):
                targets = [targets]

            # ROI 내부에 있는 객체들만 처리 (바운딩박스, ID 표시 및 객체 위치 업데이트)
            for target in targets:
                if isinstance(target, np.ndarray) and target.ndim == 2:
                    for obj in target:
                        x1, y1, x2, y2, track_id = map(int, obj[:5])
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        if is_inside_roi(center, ROI_POLYGON):
                            # 바운딩박스 및 ID 표시 (초록색)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            # ROI 내부이면 객체 위치 업데이트 및 현재 프레임 기록
                            if track_id not in object_positions:
                                object_positions[track_id] = {"start": center, "last": center}
                            else:
                                object_positions[track_id]["last"] = center
                            object_last_frame[track_id] = frame_count
                            # 이미 속도가 계산된 경우 속도 텍스트 표시
                            if track_id in object_speeds:
                                speed = object_speeds[track_id]
                                cv2.putText(frame, f"{speed:.2f} px/s", (x1, y2 + 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            # ROI를 벗어난 객체는 즉시 제거
                            if track_id in object_positions:
                                del object_positions[track_id]
                            if track_id in object_speeds:
                                del object_speeds[track_id]
                            if track_id in object_last_frame:
                                del object_last_frame[track_id]

            # 3초마다 속도 갱신 (UPDATE_INTERVAL마다)
            if frame_count - last_update_frame >= UPDATE_INTERVAL:
                log_text = f"\n🔹 {frame_count // FPS}초 시점 속도 로그:\n"
                speeds = []
                # ROI 내부에 지속적으로 감지된 객체만 고려: 마지막 업데이트 프레임이 현재 프레임이어야 함.
                for track_id, pos in list(object_positions.items()):
                    if object_last_frame.get(track_id, 0) != frame_count:
                        del object_positions[track_id]
                        if track_id in object_speeds:
                            del object_speeds[track_id]
                        continue
                    start_pos = pos["start"]
                    last_pos = pos["last"]
                    displacement = np.sqrt((last_pos[0] - start_pos[0])**2 + (last_pos[1] - start_pos[1])**2)
                    speed = (displacement / 3.0) if displacement >= MIN_DISTANCE else 0
                    object_speeds[track_id] = speed
                    speeds.append(speed)
                    log_text += f"객체 ID {track_id}: 이동 거리 {displacement:.2f} px, 속도 {speed:.2f} px/s\n"
                    # 다음 구간 계산을 위해 시작 위치를 최신 위치로 갱신
                    object_positions[track_id]["start"] = last_pos
                avg_speed = np.mean(speeds) if speeds else 0
                avg_speed_text = f"Avg Speed: {avg_speed:.2f} px/s"
                log_text += f"평균 속도: {avg_speed:.2f} px/s\n\n"
                speed_log += log_text
                last_update_frame = frame_count

            # 프레임에 평균 속도 및 FPS 표시
            cv2.putText(frame, avg_speed_text, (10, frame_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elapsed_time = time.time() - start_time
            fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 결과 프레임 저장 및 실시간 표시
            out.write(frame)
            frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        # 종료 처리
        cap.release()
        out.release()
        os.remove(video_path)
        st.success("모델 실행이 완료되었습니다!")

        # 최종 결과 계산 (ROI 직선 길이 및 예상 대기 시간)
        roi_length, _, _ = calculate_roi_mid_length(ROI_POLYGON)
        final_avg_speed = np.mean(list(object_speeds.values())) if object_speeds else 0
        if final_avg_speed > 0:
            estimated_wait_time = roi_length / final_avg_speed
        else:
            estimated_wait_time = float('inf')
        final_summary = f"최종 평균 속도: {final_avg_speed:.2f} px/s\nROI 직선 길이: {roi_length:.2f} px\n"
        if estimated_wait_time == float('inf'):
            final_summary += "예상 대기 시간: 측정 불가 (속도 0)\n"
        else:
            final_summary += f"예상 대기 시간: {estimated_wait_time:.2f} 초\n"

        # 토글(expander)로 속도 로그 및 최종 결과 표시
        with st.expander("속도 로그 및 최종 결과 보기"):
            st.text(speed_log + "\n===== 최종 결과 =====\n" + final_summary)

        # 최종 결과 비디오를 'data/output' 폴더에 저장
        final_result_folder = os.path.join("data", "output")
        os.makedirs(final_result_folder, exist_ok=True)
        final_output_path = os.path.join(final_result_folder, os.path.basename(output_video_path))
        os.rename(output_video_path, final_output_path)
        st.success(f"비디오가 {final_output_path}에 저장되었습니다!")
        
        # 최종 결과 로그를 txt 파일도 저장
        log_file_path = os.path.join(final_result_folder, f"{os.path.splitext(os.path.basename(output_video_path))[0]}_speed_log.txt")
        with open(log_file_path, "w") as f:
            f.write(speed_log + "\n===== 최종 결과 =====\n" + final_summary)
        st.success(f"속도 로그가 {log_file_path}에 저장되었습니다!")

