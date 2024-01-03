# VS_HW4

## requirements
To set up environment
```
conda env create -f environment.yml
```
Also download FFMpeg from [here](https://ffmpeg.org/download.html)


## DeepSORT
Please download my deepSORT model via Google Drive: https://drive.google.com/file/d/15sTG4Nka_JzrCweggssy701apw3bjReL/view?usp=sharing
(since our GREATEST github does not want to let me upload ^^) 

To activate camera tracking
```
cd yolo_tracking
python track.py --source 0 --show-vid
```

## TS, M3U8 Generation
you can use "--out" to specify output location
```
python track.py --source 0 --show-vid --save-vid --out inference/output
```


## JPEG streaming with ui
Put yolov5x_50.pt into yolo_tracking_streaming\yolov5\weights
Put ckpt.t7 and original_ckpt.t7 into yolo_tracking_streaming\deep_sort_pytorch\deep_sort\deep\checkpoint
```
python app.py --source video_path
```
For camera, use --source 0
