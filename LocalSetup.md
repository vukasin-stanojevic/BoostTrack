# Inference

- Added Required weights from the repo
  - SBS for embedding (fast-reid)
- Added the dataset to folder data **Note Differs from main README**
- Modified the test/train to include test: MOT20-01 only, train: removed MOT20-01
- Modified Sequence Info Accordingly
- Dataset Conversion to coco using

```
python3 data/tools/convert_mot20_to_coco.py
```

## BoostTrack Inference

**No Training for detection included**

- MOT20-01 Inference

```
python main.py --dataset mot20 --exp_name BoostTrack --no_reid --btpp_arg_iou_boost --btpp_arg_no_sb --btpp_arg_no_vt --test_dataset --detector "yoloV11"
```

by default yoloV11 is the detector, however it can be replaced with yolox.  
the default is yolox pretrained on MOT17  
to change to yolox pretrained on MOT20
change line 18 in default*settings.py to
`detector_path = "external/weights/bytetrack_x_mot20.tar"`  
\*\*\_NOTE you should have this weight in the path*\*\*

- To Get HOTA Score  
  To change the pathName assuming you delete results/trackers after each experiment

```
mv results/trackers/MOT20-val results/trackers/MOT20-test
```

To calculate HOTA score  
**_NOTE this needs to be test not val as in repository for MOT20-01_**

```
python external/TrackEval/scripts/run_mot_challenge.py   --SPLIT_TO_EVAL test   --GT_FOLDER results/gt/   --TRACKERS_FOLDER results/trackers/   --BENCHMARK MOT20  --TRACKERS_TO_EVAL BoostTrack_post_gbi
```

can run the HOTA also for \_post or just BoostTrack by replacing the last argument

**For another test**

```
rm -r cache/ results/trackers/
```

## BoostTrack+ Inference

**No Training for detection included**

- MOT20-01 Inference

```
python main.py --dataset mot20 --exp_name BoostTrackPlus --btpp_arg_iou_boost --btpp_arg_no_sb --btpp_arg_no_vt --test_dataset --detector "yoloV11"
```

by default yoloV11 is the detector, however it can be replaced with yolox.  
the default is yolox pretrained on MOT17  
to change to yolox pretrained on MOT20
change line 18 in default*settings.py to
`detector_path = "external/weights/bytetrack_x_mot20.tar"`  
\*\*\_NOTE you should have this weight in the path*\*\*

- To Get HOTA Score  
  To change the pathName assuming you delete results/trackers after each experiment

```
mv results/trackers/MOT20-val results/trackers/MOT20-test
```

To calculate HOTA score  
**_NOTE this needs to be test not val as in repository for MOT20-01_**

```
python external/TrackEval/scripts/run_mot_challenge.py   --SPLIT_TO_EVAL test   --GT_FOLDER results/gt/   --TRACKERS_FOLDER results/trackers/   --BENCHMARK MOT20  --TRACKERS_TO_EVAL BoostTrackPlus_post_gbi
```

can run the HOTA also for \_post or just BoostTrackPlus by replacing the last argument

**For another test**

```
rm -r cache/ results/trackers/
```

## BoostTrack++ Inference

**No Training for detection included**

- MOT20-01 Inference

```
python main.py --dataset mot20 --exp_name BTPP --test_dataset --detector "yoloV11"
```

by default yoloV11 is the detector, however it can be replaced with yolox.  
the default is yolox pretrained on MOT17  
to change to yolox pretrained on MOT20
change line 18 in default*settings.py to
`detector_path = "external/weights/bytetrack_x_mot20.tar"`  
\*\*\_NOTE you should have this weight in the path*\*\*

- To Get HOTA Score  
  To change the pathName assuming you delete results/trackers after each experiment

```
mv results/trackers/MOT20-val results/trackers/MOT20-test
```

To calculate HOTA score  
**_NOTE this needs to be test not val as in repository for MOT20-01_**

```
python external/TrackEval/scripts/run_mot_challenge.py   --SPLIT_TO_EVAL test   --GT_FOLDER results/gt/   --TRACKERS_FOLDER results/trackers/   --BENCHMARK MOT20  --TRACKERS_TO_EVAL BTPP_post_gbi
```

can run the HOTA also for \_post or just BoostTrack by replacing the last argument

**For another test**

```
rm -r cache/ results/trackers/
```

# Experiments

## BoostTrack

- yoloV11: 45.755
- yoloX mot17: 66.164
- yoloX mot20: 75.783

## BoostTrack+

- yoloV11: 50.241
- yoloX mot17: 68.468
- yoloX mot20: 77.499

## BoostTrack++

- yoloV11: 50.175
- yoloX mot17: 68.156
- yoloX mot20: 78.097
