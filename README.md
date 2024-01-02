# VS_HW4

## requirements
To set up environment
```
conda env create -f environment.yml
```
Also download FFMpeg from [here](https://ffmpeg.org/download.html)


## DeepSORT
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