import os
import shutil
import time
from typing import Tuple

import numpy as np

import dataset
import utils
from args import make_parser
from external.adaptors import detector
from tracker.GBI import GBInterpolation
from tracker.boost_track import BoostTrack

"""
Script modified from Deep OC-SORT: 
https://github.com/GerardMaggiolino/Deep-OC-SORT
"""


def get_detector_path_and_im_size(args) -> Tuple[str, Tuple[int, int]]:
    if args.dataset == "mot17":
        if args.test_dataset:
            detector_path = "external/weights/bytetrack_x_mot17.pth.tar"
        else:
            detector_path = "external/weights/bytetrack_ablation.pth.tar"
        size = (800, 1440)
    elif args.dataset == "mot20":
        if args.test_dataset:
            detector_path = "external/weights/bytetrack_x_mot20.tar"
            size = (896, 1600)
        else:
            # Just use the mot17 test model as the ablation model for 20
            detector_path = "external/weights/bytetrack_x_mot17.pth.tar"
            size = (800, 1440)
    else:
        raise RuntimeError("Need to update paths for detector for extra datasets.")
    return detector_path, size


def get_main_args():
    parser = make_parser()
    parser.add_argument("--dataset", type=str, default="mot17")
    parser.add_argument("--result_folder", type=str, default="results/trackers/")
    parser.add_argument("--test_dataset", action="store_true")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--lambda_iou", type=float, default=0.5)
    parser.add_argument("--lambda_mhd", type=float, default=0.25)
    parser.add_argument("--lambda_shape", type=float, default=0.25)
    parser.add_argument("--dlo_boost_coef", type=float, default=0.65)
    parser.add_argument("--det_thresh", type=float, default=0.6)
    parser.add_argument("--no_dlo", action="store_true", help="mark if detecting likely objects step should NOT be performed")
    parser.add_argument("--no_duo", action="store_true", help="mark if detecting unlikely objects step should NOT be performed")
    parser.add_argument("--no_cmc", action="store_true", help="mark if CMC should NOT be performed")
    parser.add_argument("--no_reid", action="store_true", help="mark if visual embedding should NOT be used")
    parser.add_argument("--min_box_area", type=float, default=10, help="filter out tiny boxes")
    parser.add_argument(
        "--aspect_ratio_thresh",
        type=float,
        default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value.",
    )
    parser.add_argument(
        "--no_post",
        action="store_true",
        help="do not run post-processing linear interpolation.",
    )

    args = parser.parse_args()
    if args.dataset == "mot17":
        args.result_folder = os.path.join(args.result_folder, "MOT17-val")
    elif args.dataset == "mot20":
        args.result_folder = os.path.join(args.result_folder, "MOT20-val")
    if args.test_dataset:
        args.result_folder.replace("-val", "-test")
    return args


video_to_frame_rate = {"MOT17-13-FRCNN": 25, "MOT17-11-FRCNN": 30,
                       "MOT17-10-FRCNN": 30, "MOT17-09-FRCNN": 30,
                       "MOT17-05-FRCNN": 14, "MOT17-04-FRCNN": 30,
                       "MOT17-02-FRCNN": 30, "MOT20-05": 25,
                       "MOT20-03": 25, "MOT20-02": 25, "MOT20-01": 25,
                        "MOT17-14-SDP":	25, "MOT17-12-SDP":	30,
                        "MOT17-08-SDP":	30, "MOT17-07-SDP":	30,
                        "MOT17-06-SDP":	14, "MOT17-03-SDP":	30,
                        "MOT17-01-SDP":	30, "MOT17-14-FRCNN": 25,
                        "MOT17-12-FRCNN": 30, "MOT17-08-FRCNN": 30,
                        "MOT17-07-FRCNN": 30, "MOT17-06-FRCNN": 14,
                        "MOT17-03-FRCNN": 30, "MOT17-01-FRCNN": 30,
                        "MOT17-14-DPM": 25, "MOT17-12-DPM":	30,
                        "MOT17-08-DPM":	30, "MOT17-07-DPM":	30,
                        "MOT17-06-DPM":	14, "MOT17-03-DPM":	30, "MOT17-01-DPM":	30,
                       "MOT20-08": 25, "MOT20-07": 25, "MOT20-06": 25, "MOT20-04": 25
                       }

for k in video_to_frame_rate:
    video_to_frame_rate[k] = max(int(video_to_frame_rate[k] * 2), 30)


def main():
    # Set dataset and detector
    args = get_main_args()

    detector_path, size = get_detector_path_and_im_size(args)
    det = detector.Detector("yolox", detector_path, args.dataset)
    loader = dataset.get_mot_loader(args.dataset, args.test_dataset, size=size)

    use_ecc = not args.no_cmc
    use_embedding = not args.no_reid

    lambda_iou = args.lambda_iou
    lambda_mhd = args.lambda_mhd
    lambda_shape = args.lambda_shape
    use_dlo_boost = not args.no_dlo
    use_duo_boost = not args.no_duo

    tracker = None
    results = {}
    frame_count = 0
    total_time = 0
    # See __getitem__ of dataset.MOTDataset
    for (img, np_img), label, info, idx in loader:
        # Frame info
        frame_id = info[2].item()
        video_name = info[4][0].split("/")[0]

         # Hacky way to skip SDP and DPM when testing
        if "FRCNN" not in video_name and args.dataset == "mot17":
            continue
        tag = f"{video_name}:{frame_id}"
        if video_name not in results:
            results[video_name] = []

        img = img.cuda()

        # Initialize tracker on first frame of a new video
        print(f"Processing {video_name}:{frame_id}\r", end="")
        if frame_id == 1:
            print(f"Initializing tracker for {video_name}")
            print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
            if tracker is not None:
                tracker.dump_cache()

            tracker = BoostTrack(det_thresh=args.det_thresh,
                                use_ecc=use_ecc,
                                use_embedding=use_embedding,
                                lambda_iou=lambda_iou,
                                lambda_mhd=lambda_mhd,
                                lambda_shape=lambda_shape,
                                dlo_boost_coef=args.dlo_boost_coef,
                                use_dlo_boost=use_dlo_boost,
                                use_duo_boost=use_duo_boost,
                                max_age=video_to_frame_rate[video_name],
                                video_name=video_name,
                                dataset_name=args.dataset,
                                test_dataset=args.test_dataset)

        pred = det(img, tag)
        start_time = time.time()

        if pred is None:
            continue
        # Nx5 of (x1, y1, x2, y2, ID)
        targets = tracker.update(pred, img, np_img[0].numpy(), tag)
        tlwhs, ids, confs = utils.filter_targets(targets, args.aspect_ratio_thresh, args.min_box_area)

        total_time += time.time() - start_time
        frame_count += 1

        results[video_name].append((frame_id, tlwhs, ids, confs))

    print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")

    # Save detector results
    det.dump_cache()
    tracker.dump_cache()
    # Save for all sequences
    folder = os.path.join(args.result_folder, args.exp_name, "data")
    os.makedirs(folder, exist_ok=True)
    for name, res in results.items():
        result_filename = os.path.join(folder, f"{name}.txt")
        utils.write_results_no_score(result_filename, res)
    print(f"Finished, results saved to {folder}")

    # args.no_post = True
    if not args.no_post:
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

        # import os
        if not os.path.exists(post_folder_gbi):
            os.makedirs(post_folder_gbi)
        for file_name in os.listdir(res_folder):
            in_path = os.path.join(post_folder_data, file_name)
            out_path2 = os.path.join(post_folder_gbi, file_name)

            GBInterpolation(
                path_in=in_path,
                path_out=out_path2,
                interval=interval,
                tau=10.5
            )
        print(f"Gradient boosting interpolation post-processing applied, saved to {post_folder_gbi}.")


if __name__ == "__main__":
    main()
