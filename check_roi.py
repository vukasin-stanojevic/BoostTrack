import cv2
import os
import numpy as np

# 글로벌 변수 설정
points = []  # 다각형 꼭지점 리스트
img = None   # 현재 이미지(또는 프레임) 저장 변수

def draw_polygon(event, x, y, flags, param):
    """ 마우스로 꼭지점 추가 및 이미지 업데이트 """
    global points, img

    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 클릭(꼭지점 추가)
        points.append((x, y))
        redraw_image()

def redraw_image():
    """ 선택한 꼭지점과 다각형을 다시 그림 """
    temp_img = img.copy()
    
    # 모든 꼭지점 그리기
    for point in points:
        cv2.circle(temp_img, point, 5, (0, 255, 0), -1)
    
    # 꼭지점이 3개 이상일 때 닫힌 다각형을 그림
    if len(points) > 2:
        cv2.polylines(temp_img, [np.array(points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
    
    cv2.imshow("Image", temp_img)

def process_media():
    """ 이미지 및 동영상 파일을 하나씩 표시하고 다각형 선택 """
    global img, points

    folder_path = "data/videos"  # 파일이 들어있는 폴더 지정
    if not os.path.exists(folder_path):
        print(f"폴더 '{folder_path}'가 존재하지 않습니다.")
        return

    image_extensions = ('.jpg', '.jpeg', '.png')
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')

    media_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions + video_extensions)]
    if not media_files:
        print(f"'{folder_path}' 폴더에 이미지 또는 동영상 파일이 없습니다.")
        return

    for media_file in media_files:
        media_path = os.path.join(folder_path, media_file)
        
        # 이미지 파일인 경우
        if media_file.lower().endswith(image_extensions):
            img = cv2.imread(media_path)
            if img is None:
                print(f"이미지를 로드할 수 없음: {media_path}")
                continue
        # 동영상 파일인 경우, 첫 프레임을 읽어옴
        elif media_file.lower().endswith(video_extensions):
            cap = cv2.VideoCapture(media_path)
            ret, frame = cap.read()  # 첫 프레임 읽기
            cap.release()
            if not ret:
                print(f"동영상에서 프레임을 가져올 수 없음: {media_path}")
                continue
            img = frame

        points = []  # 꼭지점 초기화

        cv2.imshow("Image", img)
        cv2.setMouseCallback("Image", draw_polygon)

        print(f"'{media_file}'에서 다각형 꼭지점 선택 후 'n' 키를 눌러 다음 파일로 이동하세요.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):  # 'n' 키를 누르면 다음 파일로 이동
                print(f"{media_file}의 다각형 꼭지점 좌표: {points}")
                break
            elif key == ord('q'):  # 'q' 키를 누르면 종료
                print("프로그램 종료")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_media()
