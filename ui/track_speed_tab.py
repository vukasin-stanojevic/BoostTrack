import streamlit as st
import cv2
import torch
import tempfile
import os
import numpy as np
import time
from external.adaptors.detector import Detector
from tracker.boost_track import BoostTrack
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from ui.utils.roi import extract_thumbnail, is_inside_roi, calculate_roi_mid_length, roi_ui

ROI_POLYGON = np.array([(613, 19), (489, 37), (572, 329), (718, 296)], np.int32)
FPS = 30                   # 비디오 FPS (기본값, 실제 FPS는 비디오에서 가져옴)
UPDATE_INTERVAL = 3 * FPS  # 3초마다 속도 갱신 (프레임 단위)
MIN_DISTANCE = 5.0         # 최소 이동거리 (5px 미만은 0으로 처리)

RESIZED_RATIO = 1 / 3
DRAWING_MODE = "rect"  
STROKE_WIDTH = 3
STROKE_COLOR = "#0000FF"
FILL_COLOR = "rgba(255, 255, 255, 0.0)"

def track_tab():
    roi_ui()

    if st.button("모델 실행"):
        detector = Detector("yolox", "external/weights/bytetrack_x_mot20.tar", "custom")
        tracker = BoostTrack()

        cap = cv2.VideoCapture(st.session_video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 결과 비디오 저장 설정 -> 임시 폴더에 저장 후, 나중에 data/output으로 이동
        output_video_path = os.path.join(tempfile.gettempdir(), "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

        frame_display = st.empty()

        frame_count = 0
        last_update_frame = 0
        object_positions = {}   # {object_id: {"start": (x, y), "last": (x, y)}}
        object_speeds = {}      # {object_id: speed (px/s)}
        object_last_frame = {}  # {object_id: 마지막 업데이트된 frame 번호}
        avg_speed_text = "Avg Speed: 0.00 px/s"
        speed_log = ""          # 속도 로그를 누적해서 저장할 문자열

        # ROI 직선 길이 및 중간 선 좌표 계산 -> 대기열 길이
        roi_length, left_mid, right_mid = calculate_roi_mid_length(st.session_state.roi_polygon)

        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # ROI 영역 그리기 (ROI 다각형과 ROI 중간 선(파란색) 표시)
            cv2.polylines(frame, [st.session_state.roi_polygon], isClosed=True, color=(255, 0, 0), thickness=2)
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
                        if is_inside_roi(center, st.session_state.roi_polygon):
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
                # ROI 내부에 지속적으로 감지된 객   체만 고려: 마지막 업데이트 프레임이 현재 프레임이어야 함.
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

        cap.release()
        out.release()
        # os.remove(video_path)
        os.unlink(st.session_video_path)

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
