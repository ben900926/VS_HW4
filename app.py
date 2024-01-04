from flask import Flask, render_template, Response, request
import json
import cv2

import logging
import argparse
from utils.utils import *
from utils.log import logger
from utils.timer import Timer
from utils.parse_config import parse_model_cfg
import utils.datasets as datasets
from track import yield_frame


logger.setLevel(logging.INFO)

app = Flask(__name__)


## main
parser = argparse.ArgumentParser(prog='demo.py')
parser.add_argument('--cfg', type=str, default='cfg/yolov3_576x320.cfg', help='cfg file path')
parser.add_argument('--weights', type=str, default='weights/jde_576_320.pt', help='path to weights file')
parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
parser.add_argument('--input-video', type=str, help='path to the input video')
parser.add_argument('--output-format', type=str, default='video', choices=['video', 'text'], help='Expected output format. Video or text.')
parser.add_argument('--output-root', type=str, default='results', help='expected output root path')
opt = parser.parse_args()
print(opt, end='\n\n')

## track.py
result_root = opt.output_root if opt.output_root!='' else '.'
mkdir_if_missing(result_root)

cfg_dict = parse_model_cfg(opt.cfg)
opt.img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]

# run tracking
timer = Timer()
accs = []
n_frame = 0

logger.info('Starting tracking...')
# dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
dataloader = datasets.LoadStreams(opt.input_video, opt.img_size)

result_filename = os.path.join(result_root, 'results.txt')
frame_rate = dataloader.frame_rate 

frame_dir = None if opt.output_format=='text' else osp.join(result_root, 'frame')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    try:
        return Response(yield_frame(opt, dataloader, 'mot', result_filename,
                save_dir=frame_dir, show_image=False, frame_rate=frame_rate), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.info(e)


@app.route('/data', methods=["GET", "POST"])
def mouse_click_pos():
    global deepsort

    data = json.loads(request.data)
    # x, y: relative coordinate of the click on video
    x, y = data.get('x'), data.get('y')

    # TODO: do something with the track here
    if x is not None and y is not None:
        deepsort.update_display(x, y)
    
    # You can return a response if necessary
    return 'success'


if __name__ == '__main__':
    app.run()