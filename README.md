# BoostTrack

> [**BoostTrack: Boosting the similarity measure and detection confidence for improved multiple object tracking**](https://link)
> 
> Vukasin Stanojevic, Branimir Todorovic

 
<p align="center"><img src="assets/hotaplt.png" width="500"/><br>HOTA score on MOT17 and MOT20 datasets.</p>

## Abstract
Handling unreliable detections and avoiding identity switches are crucial for the success of multiple object tracking (MOT). Ideally, MOT algorithm should use true positive detections only, work in real-time and produce no identity switches. To approach the described ideal solution, we present the BoostTrack, a simple yet effective tracing-by-detection MOT method that utilizes several lightweight plug and play additions to improve MOT performance. We design a detection-tracklet confidence score and use it to scale the similarity measure and implicitly favour high detection confidence and high tracklet confidence pairs in one-stage association. To reduce the ambiguity arising from using intersection over union (IoU), we propose a novel Mahalanobis distance and shape similarity additions to boost the overall similarity measure. To utilize low-detection score bounding boxes in one-stage association, we propose to boost the confidence scores of two groups of detections:  the detections we assume to correspond to the existing tracked object, and the detections we assume to correspond to a previously undetected object. The proposed additions are orthogonal to the existing approaches, and we combine them with interpolation and camera motion compensation to achieve results comparable to the standard benchmark solutions while retaining real-time execution speed. When combined with appearance similarity, our method outperforms all standard benchmark solutions on MOT17 and MOT20 datasets. It ranks first among online methods in HOTA metric in the MOT Challenge on MOT17 and MOT20 test sets. 
<p align="center"><img src="assets/overview.png" width="600"/></p>

## Tracking performance
### Results on MOT17 test set
| Method      | HOTA   |  MOTA  |  IDF1  |  IDSW  |
|-------------|--------|--------|--------|--------|
| BoostTrack  |  65.4   |  80.5  |  80.2  | 1104 |
| BoostTrack+ |  66.4   |  80.6  |  81.8  | 1086 |

### Results on MOT20 test set
| Method      | HOTA   |  MOTA  |  IDF1  |  IDSW  |
|-------------|--------|--------|--------|--------|
|BoostTrack   | 63   | 76.4 | 76.5 | 992 |
|BoostTrack+  | 66.2 | 77.2 | 81.5 | 899 |

## Installation
We tested the code on Ubuntu 22.04.
**Step 1.** Download repository and setup the conda environment.

Note: g++ is required to install all the requirements.
```shell
gh repo clone vukasin-stanojevic/BoostTrack
cd BoostTrack
conda env create -f boost-track-env.yml
conda activate boostTrack
```
Due to numpy version error, single line of code in mapping.py file from onnx module should be modified. The line 25 of the file should be replaced with the line:
```python
    int(TensorProto.STRING): np.dtype(object)
```

**Step 2.** Download the model weights and setup the datasets.

We use the same weights as [Deep OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT/tree/main). The weights can be downloaded from the [link](https://drive.google.com/drive/folders/15hZcR4bW_Z9hEaXXjeWhQl_jwRKllauG?usp=sharing).

*2.1.* Download the weights and place to BoostTrack/external/weights folder.

*2.2.* Download MOT17 and MOT20 datasets from the [MOT Challenge website](https://motchallenge.net/).

*2.3.* Place the files under BoostTrack/data folder:
```
data
|——————MOT17
|        └——————train
|        └——————test
|——————MOT20
|        └——————train
|        └——————test
```
*2.4.* Run:

```shell
python3 data/tools/convert_mot17_to_coco.py
python3 data/tools/convert_mot20_to_coco.py
```
## Training and evaluation
### Train and evaluate BoostTrack
To train the BoostTrack on MOT17 and MOT20 validation sets run the following:
```shell
python main.py --dataset mot17 --exp_name BoostTrack --det_thresh 0.6 --dlo_boost_coef 0.65 --no_reid
python main.py --dataset mot20 --exp_name BoostTrack --det_thresh 0.4 --dlo_boost_coef 0.5 --no_reid
```
Note, three resulting folders will be created for each experiment: BoostTrack, BoostTrack_post, BoostTrack_post_gbi. The folders with the suffixes correspond to results with applied linear and gradient boosting interpolation. 

To evaluate the results using TrackEval run:
```shell
python external/TrackEval/scripts/run_mot_challenge.py   --SPLIT_TO_EVAL val   --GT_FOLDER results/gt/   --TRACKERS_FOLDER results/trackers/   --BENCHMARK MOT17  --TRACKERS_TO_EVAL BoostTrack_post_gbi
python external/TrackEval/scripts/run_mot_challenge.py   --SPLIT_TO_EVAL val   --GT_FOLDER results/gt/   --TRACKERS_FOLDER results/trackers/   --BENCHMARK MOT20  --TRACKERS_TO_EVAL BoostTrack_post_gbi 
```
### Train and evaluate BoostTrack+
Similarly, to train the BoostTrack+ run:
```shell
python main.py --dataset mot17 --exp_name BoostTrackPlus --det_thresh 0.6 --dlo_boost_coef 0.65
python main.py --dataset mot20 --exp_name BoostTrackPlus --det_thresh 0.4 --dlo_boost_coef 0.5
```
To evaluate the BoostTrack+ results run:
```shell
python external/TrackEval/scripts/run_mot_challenge.py   --SPLIT_TO_EVAL val   --GT_FOLDER results/gt/   --TRACKERS_FOLDER results/trackers/   --BENCHMARK MOT17  --TRACKERS_TO_EVAL BoostTrackPlus_post_gbi
python external/TrackEval/scripts/run_mot_challenge.py   --SPLIT_TO_EVAL val   --GT_FOLDER results/gt/   --TRACKERS_FOLDER results/trackers/   --BENCHMARK MOT20  --TRACKERS_TO_EVAL BoostTrackPlus_post_gbi
```
# Acknowledgements
Our implementation is developed on top of publicly available codes. We thank authors of [Deep OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT/), [SORT](https://github.com/abewley/sort), [StrongSort](https://github.com/dyhBUPT/StrongSORT), [NCT](https://github.com/Autoyou/Noise-control-multi-object-tracking), [ByteTrack](https://github.com/ifzhang/ByteTrack/) for making their code available. 

# Citation

If you find our work useful, please cite our paper: 
```
@article{stanojevic2024boostTrack,
  title={BoostTrack: boosting the similarity measure and detection confidence for improved multiple object tracking},
  author={Vukasin Stanojevic and Branimir Todorovic},
  journal={Machine Vision and Applications},
  year={2024},
  doi={10.1007/s00138-024-01531-5}
}
```
