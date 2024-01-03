# VS_HW4

To set up environment
```
conda env create -f environment.yml
```

To activate camera tracking
```
cd yolo_tracking
python track.py --source 0 --show-vid
```

## install falsk
##### requirement
* Python >= 3.8
```
pip install flask
```

##### Command
```
flask --app filename run   ## run filename.py
flask run                  ## run app.py
```

##### Structure
YourProject \
├─ app.py \
├─ template \
│  └─ index.html \
└─ static \
&emsp;  &nbsp;  └─ img or mp4... 


