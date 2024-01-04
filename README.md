# VS_Hw4 (JDE)

uploaded this HW with JDE model (https://github.com/Zhongdao/Towards-Realtime-MOT)

## To run app
Set up environment
```
conda env create -f environment.yml
```

also download "jde_576x320.pt" from here
(https://github.com/Zhongdao/Towards-Realtime-MOT)

Run app.py on localhost:5000
```
python app.py --input-video 0
```
This should show the live tracking results with fps=4 (roughly)

## TODO..
I have not updated the corresponding part for "Mouse Clicking" yet...