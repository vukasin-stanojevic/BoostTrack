import argparse


def make_parser():
    """ ArgumentParser 객체를 생성하여 반환하는 함수 """
    parser = argparse.ArgumentParser("BoostTrack parameters")

    # distributed
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size") # 배치 크기 (한 번에 처리할 프레임 수, 기본값 = 1)
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training") # 모델을 실행할 GPU 번호 지정 (예: 0 → GPU 0 사용)

    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training") # 분산 학습(distributed training)에서 각 GPU의 랭크(순번)
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training") # 분산 학습에서 사용할 서버(노드) 개수
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training") # 분산 학습에서 각 노드(서버)의 랭크(순번)

    # 실험을 실행할 때 사용할 설정 파일
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    
    # 테스트 모드로 실행 (이 옵션을 추가하면 args.test = True)
    # --test 모드는 테스트 데이터셋에서 평가(Evaluation)를 수행하는 모드로 추적기의 성능을 측정할 때 사용되며, 실제 영상에서 실시간으로 추적하는 것이 아니라 미리 정의된 데이터셋에서 성능을 평가하는 기능을 수행
    parser.add_argument(
        "--test",
        dest="test",
        default=False, #기본적으로 비활설화
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # det args (탐지기 관련 설정)
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval") # YOLOX 탐지기의 사전 학습된 모델 체크포인트 (weights) 경로
    parser.add_argument("--conf", default=0.1, type=float, help="test conf") # 탐지 신뢰도 임계값 (default=0.1, 즉 10% 이상 신뢰도일 때 탐지된 객체로 인정)
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold") # NMS(Non-Maximum Suppression) 임계값 (default=0.7) -> --nms를 높이면 유사한 바운딩 박스를 더 많이 제거할 수 있음.
    parser.add_argument("--tsize", default=[800, 1440], nargs="+", type=int, help="test img size") # 입력 이미지 크기 (기본값 [800, 1440])
    parser.add_argument("--seed", default=None, type=int, help="eval seed") # 재현성을 위한 난수 시드 설정

    # tracking args (추적기 관련 설정)
    # --track_thresh와 --iou_thresh 값을 조정하면 추적 성능을 최적화 가능
    parser.add_argument("--track_thresh", type=float, default=0.6, help="detection confidence threshold") # 탐지 신뢰도 임계값 (기본값 0.6, 즉 60% 이상 신뢰도일 때만 추적)
    parser.add_argument( # IoU(Intersection over Union) 임계값 (기본값 0.3, 즉 IoU가 0.3 이상이면 같은 객체로 판단
        "--iou_thresh",
        type=float,
        default=0.3,
        help="the iou threshold in Sort for matching",
    )
    parser.add_argument("--min_hits", type=int, default=3, help="min hits to create track in SORT") # 최소 프레임 수 (객체가 추적 리스트에 유지되려면 최소 3번 이상 탐지되어야 함)
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks") # 사라진 객체를 추적 상태로 유지할 최대 프레임 수
    parser.add_argument( # 객체 매칭을 위한 임계값 (default=0.9)
        "--match_thresh",
        type=float,
        default=0.9,
        help="matching threshold for tracking",
    )
    parser.add_argument(
        "--gt-type",
        type=str,
        default="_val_half",
        help="suffix to find the gt annotation",
    )
    parser.add_argument("--public", action="store_true", help="use public detection")

    # for kitti/bdd100k inference with public detections
    parser.add_argument(
        "--raw_results_path",
        type=str,
        default="exps/permatrack_kitti_test/",
        help="path to the raw tracking results from other tracks",
    )
    parser.add_argument("--out_path", type=str, help="path to save output results")
    parser.add_argument(
        "--hp",
        action="store_true",
        help="use head padding to add the missing objects during \
            initializing the tracks (offline).",
    )

    # for demo video(비디오 및 웹캠 설정)
    parser.add_argument("--demo_type", default="image", help="demo type, eg. image, video and webcam") # 실행 모드 (image, video, webcam 중 선택)
    parser.add_argument("--path", default="./videos/demo.mp4", help="path to images or video") # 비디오 파일 경로 (기본값: ./videos/demo.mp4)
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id") # 웹캠 사용 시 카메라 ID (기본값: 0, 즉 기본 웹캠 사용)
    parser.add_argument( # 결과 저장 옵션 (True 설정 시 탐지 결과를 저장)
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument( # --device: gpu 또는 cpu 설정
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    return parser
