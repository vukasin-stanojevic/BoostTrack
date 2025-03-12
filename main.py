import os
import shutil
import time

import dataset
import utils
from args import make_parser
from default_settings import GeneralSettings, get_detector_path_and_im_size, BoostTrackPlusPlusSettings, BoostTrackSettings
from external.adaptors import detector
from tracker.GBI import GBInterpolation
from tracker.boost_track import BoostTrack

"""
Script modified from Deep OC-SORT: 
https://github.com/GerardMaggiolino/Deep-OC-SORT
"""

# 입력 인자 설정
def get_main_args():
    # parser는 명령줄 인자를 해석하는 객체
    parser = make_parser() # make_parser() 함수가 ArgumentParser 객체를 생성하고 parser에 저장
    parser.add_argument("--result_folder", type=str, default="results/trackers/")
    parser.add_argument("--test_dataset", action="store_true")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--no_reid", action="store_true", help="mark if visual embedding should NOT be used") #Appearance Similarity (ReID) 기능 비활성화
    parser.add_argument("--no_cmc", action="store_true", help="mark if camera motion compensation should NOT be used")

    parser.add_argument("--s_sim_corr", action="store_true", help="mark if you want to use corrected version of shape similarity calculation function")

    parser.add_argument("--btpp_arg_iou_boost", action="store_true", help="BoostTrack++ arg. Mark if only IoU should be used for detection confidence boost.")
    parser.add_argument("--btpp_arg_no_sb", action="store_true", help="BoostTrack++ arg. Mark if soft detection confidence boost should NOT be used.")
    parser.add_argument("--btpp_arg_no_vt", action="store_true", help="BoostTrack++ arg. Mark if varying threhold should NOT be used for the detection confidence boost.")

    parser.add_argument(
        "--no_post",
        action="store_true",
        help="do not run post-processing.",
    )

    # args.dataset 값이 "mot17"이면 결과 저장 폴더를 "MOT17-val"로 지정
    # argparse 라이브러리를 사용하여 args 객체를 생성 (args는 parser.parse_args()를 통해 사용자의 명령줄 입력을 저장한 객체)
    # MOTChallenge 데이터셋을 사용할 때 자동으로 결과 저장 경로를 지정하는 코드 -> 커스텀 데이터셋을 사용하려면 이 부분을 수정해야 함
    args = parser.parse_args() # 명령줄 인자 파싱하여 args에 저장
    if args.dataset == "mot17":
        args.result_folder = os.path.join(args.result_folder, "MOT17-val")
    elif args.dataset == "mot20":
        args.result_folder = os.path.join(args.result_folder, "MOT20-val")
    elif args.dataset == "custom":
        args.result_folder = os.path.join(args.result_foler, "custom_dataset")

    if args.test_dataset:
        args.result_folder.replace("-val", "-test")
    return args


def main():
    # 실시간 실행하려면 --test 없이 실행해야 함.
    # Set dataset and detector (탐지기 및 데이터 로더 초기화)
    args = get_main_args() # 사용자가 실행할 때 입력한 명령줄 옵션을 args 객체에 저장
    GeneralSettings.values['dataset'] = args.dataset # 사용자가 선택한 데이터셋을 저장 (예: "mot17", "mot20", "custom")
    GeneralSettings.values['use_embedding'] = not args.no_reid
    GeneralSettings.values['use_ecc'] = not args.no_cmc
    GeneralSettings.values['test_dataset'] = args.test_dataset

    BoostTrackSettings.values['s_sim_corr'] = args.s_sim_corr #BoostTrack에서 객체 형태 유사도를 조정하는 파라미터 -> 객체의 형태 정보를 비교할 때 사용할 유사도 계산 방식의 파라미터를 설정하는 부분

    BoostTrackPlusPlusSettings.values['use_rich_s'] = not args.btpp_arg_iou_boost # IOU 부스팅 사용 여부
    BoostTrackPlusPlusSettings.values['use_sb'] = not args.btpp_arg_no_sb # Soft Boosting 사용 여부
    BoostTrackPlusPlusSettings.values['use_vt'] = not args.btpp_arg_no_vt # Varying Threshold(가변 임계값) 사용 여부

    detector_path, size = get_detector_path_and_im_size(args) # YOLOX 모델의 경로(detector_path)와 입력 이미지 크기(size)를 결정
    # "yolox" 모델을 사용하며, 사전 학습된 가중치(Weights)를 detector_path에서 로드
    det = detector.Detector("yolox", detector_path, args.dataset) # Detector 클래스의 인스턴스를 생성하여 det에 저장
    loader = dataset.get_mot_loader(args.dataset, args.test_dataset, size=size) # 데이터셋을 로드하여 비디오 프레임을 하나씩 가져옴

    tracker = None #BoostTrack++ 추적기를 처음에는 생성하지 않음 (첫 번째 프레임에서 초기화할 예정)
    results = {} # 각 비디오 시퀀스의 추적 결과를 저장할 딕셔너리
    frame_count = 0 # 현재까지 처리한 프레임 수
    total_time = 0 #  전체 프레임을 처리하는 데 걸린 시간
    
    # See __getitem__ of dataset.MOTDataset
    for (img, np_img), label, info, idx in loader: # 비디오에서 한 프레임씩 읽어옴
        # Frame info
        frame_id = info[2].item() # 현재 프레임 번호
        video_name = info[4][0].split("/")[0] # 현재 비디오 파일의 이름

        # Hacky way to skip SDP and DPM when testing
        # MOT17 데이터셋에는 여러 탐지기(FRCNN, DPM, SDP)가 포함되어 있음
        if "FRCNN" not in video_name and args.dataset == "mot17": # MOT17에서 FRCNN 기반 데이터가 아니라면 탐지를 수행하지 않음
            continue
        
        # YOLOX 탐지기(det(img, tag))와 BoostTrack++ 추적기(tracker.update(...))에서 현재 프레임을 구별할 수 있도록 태그를 생성
        # 특정 프레임에서 오류가 발생하면 어떤 비디오 파일의 몇 번째 프레임인지 쉽게 디버깅할 수 있음
        tag = f"{video_name}:{frame_id}"
        
        # results는 각 비디오별로 추적된 객체의 결과를 저장하는 딕셔너리
        if video_name not in results:
            results[video_name] = []

        # 현재 프레임의 이미지 데이터를 담고 있는 텐서(Tensor)
        img = img.cuda() # 이미지를 GPU로 이동하여 연산 속도를 빠르게 함

        # Initialize tracker on first frame of a new video
        # 비디오가 변경될 때마다 새로운 Tracker를 생성하여 ID를 관리
        print(f"Processing {video_name}:{frame_id}\r", end="")
        if frame_id == 1: # 새로운 비디오의 첫 번째 프레임(frame_id == 1)이면 새로운 Tracker를 생성
            print(f"Initializing tracker for {video_name}")
            print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
            if tracker is not None:
                tracker.dump_cache() # 이전 비디오의 추적 데이터를 저장

            tracker = BoostTrack(video_name=video_name) # BoostTrack++ 추적기 초기화.

        pred = det(img, tag) # YOLOX 탐지기를 사용하여 현재 프레임에서 객체 탐지 수행
        start_time = time.time() # 탐지 및 추적 실행 시간을 측정하기 위해 현재 시간을 저장

        if pred is None: # 현재 프레임에서 탐지된 객체가 없으면 (pred is None) 추적을 수행하지 않음
            continue
        # Nx5 of (x1, y1, x2, y2, ID)
        targets = tracker.update(pred, img, np_img[0].numpy(), tag) # BoostTrack++을 사용하여 탐지된 객체를 추적 -> targets는 업데이트된 객체 목록 (객체 ID, 바운딩 박스, 신뢰도 등 포함)
        # utils.filter_targets() → 너무 작은 객체나 비정상적인 바운딩 박스를 필터링ㅇㅇ
        tlwhs, ids, confs = utils.filter_targets(targets, GeneralSettings['aspect_ratio_thresh'], GeneralSettings['min_box_area'])

        total_time += time.time() - start_time # 프레임을 처리하는 데 걸린 시간 누적
        frame_count += 1

        results[video_name].append((frame_id, tlwhs, ids, confs)) # 현재 프레임의 객체 정보를 저장

    print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
    print(total_time)
    # Save detector results
    # 탐지 및 추적 데이터를 디스크에 저장하여 나중에 사용할 수 있도록 함
    det.dump_cache()
    tracker.dump_cache()
    # Save for all sequences
    folder = os.path.join(args.result_folder, args.exp_name, "data")
    os.makedirs(folder, exist_ok=True)
    for name, res in results.items():
        result_filename = os.path.join(folder, f"{name}.txt") # 탐지 및 추적 결과를 results 딕셔너리에서 name.txt 파일로 저장
        utils.write_results_no_score(result_filename, res) #결과 저장 및 후처리 (Linear Interpolation → 끊어진 객체 트랙을 보완, Gradient Boosting Interpolation (GBI) → 객체 추적 성능을 개선)
    print(f"Finished, results saved to {folder}")
    if not args.no_post: #  사용자가 --no_post 옵션을 주지 않으면 자동으로 후처리를 수행함
        post_folder = os.path.join(args.result_folder, args.exp_name + "_post")
        pre_folder = os.path.join(args.result_folder, args.exp_name)
        if os.path.exists(post_folder):
            print(f"Overwriting previous results in {post_folder}")
            shutil.rmtree(post_folder)
        shutil.copytree(pre_folder, post_folder)
        post_folder_data = os.path.join(post_folder, "data")
        interval = 1000  # i.e. no max interval
        utils.dti(post_folder_data, post_folder_data, n_dti=interval, n_min=25)

        print(f"Linear interpolation post-processing applied, saved to {post_folder_data}.")

        res_folder = os.path.join(args.result_folder, args.exp_name, "data")
        post_folder_gbi = os.path.join(args.result_folder, args.exp_name + "_post_gbi", "data")

        if not os.path.exists(post_folder_gbi):
            os.makedirs(post_folder_gbi)
        for file_name in os.listdir(res_folder):
            in_path = os.path.join(post_folder_data, file_name)
            out_path2 = os.path.join(post_folder_gbi, file_name)

            GBInterpolation(
                path_in=in_path,
                path_out=out_path2,
                interval=interval
            )
        print(f"Gradient boosting interpolation post-processing applied, saved to {post_folder_gbi}.")


if __name__ == "__main__":
    main()
