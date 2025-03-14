import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

RESIZED_RATIO = 1 / 3
DRAWING_MODE = "rect"  
STROKE_WIDTH = 3
STROKE_COLOR = "#0000FF"
FILL_COLOR = "rgba(255, 255, 255, 0.0)"
ROI_POLYGON = np.array([(613, 19), (489, 37), (572, 329), (718, 296)], np.int32)

def extract_thumbnail(video_path, resized_ratio=RESIZED_RATIO):
    """비디오에서 첫 프레임을 추출하고 리사이징합니다."""
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    
    if not success:
        st.error("영상 파일을 읽을 수 없습니다.")
        return None, None, None
    
    original_height, original_width = frame.shape[:2]
    
    new_width = int(original_width * resized_ratio)
    new_height = int(original_height * resized_ratio)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    return resized_frame_rgb, original_width, original_height

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

def roi_ui():
    if 'thumbnail' not in st.session_state:
        st.session_state.thumbnail = None
    if 'thumbnail_array' not in st.session_state:
        st.session_state.thumbnail_array = None
    if 'canvas_result' not in st.session_state:
        st.session_state.canvas_result = None
    if 'roi_polygon' not in st.session_state:
        st.session_state.roi_polygon = ROI_POLYGON 
    if 'video_path' not in st.session_state:
        st.session_video_path = None

    uploaded_file = st.file_uploader("비디오 파일을 업로드하세요", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            # video_path = tmp_file.name
            st.session_video_path = tmp_file.name

        st.video(uploaded_file)  

        resized_frame, original_width, original_height = extract_thumbnail(st.session_video_path)
        if resized_frame is not None:
            thumbnail = Image.fromarray(resized_frame)
            st.session_state.thumbnail = thumbnail
            st.session_state.thumbnail_array = resized_frame
            
            st.info(f"원본 크기: {original_width} x {original_height} | 리사이즈 후: {thumbnail.width} x {thumbnail.height}")
            
            img_height, img_width = resized_frame.shape[:2]
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("사각형 그려")
                canvas_result = st_canvas(
                    fill_color=FILL_COLOR,
                    stroke_width=STROKE_WIDTH,
                    stroke_color=STROKE_COLOR,
                    background_image=thumbnail,
                    drawing_mode=DRAWING_MODE,
                    key="canvas",
                    width=img_width,
                    height=img_height,
                    update_streamlit=True,
                )
                
                if canvas_result.json_data is not None:
                    st.session_state.canvas_result = canvas_result.json_data
                    
                    if 'objects' in canvas_result.json_data:
                        num_rects = len(canvas_result.json_data['objects'])
                        st.write(f"그린 사각형 개수: {num_rects}개")
            
            with col2:
                st.subheader("사각형 좌표")
                
                if st.button("첫 번째 사각형 좌표 저장"):
                    if (st.session_state.canvas_result and 
                        'objects' in st.session_state.canvas_result and 
                        len(st.session_state.canvas_result['objects']) > 0):
                        
                        first_rect = st.session_state.canvas_result['objects'][0]
                        left = int(first_rect['left'])
                        top = int(first_rect['top'])
                        width = int(first_rect['width'])
                        height = int(first_rect['height'])

                        original_left = int(left / RESIZED_RATIO)
                        original_top = int(top / RESIZED_RATIO)
                        original_width_rect = int(width / RESIZED_RATIO)
                        original_height_rect = int(height / RESIZED_RATIO)

                        st.session_state.roi_polygon = np.array([
                            (original_left, original_top),
                            (original_left + original_width_rect, original_top),
                            (original_left + original_width_rect, original_top + original_height_rect),
                            (original_left, original_top + original_height_rect)
                        ], np.int32)

                        st.success(f"첫 번째 사각형 좌표가 저장되었습니다!")

                if st.session_state.roi_polygon is not None:
                    st.write("✅ **저장된 ROI 좌표:**")
                    st.code(f"{st.session_state.roi_polygon}")

        else:
            st.error("영상 파일을 읽을 수 없습니다. 다른 파일을 시도해 보세요.")