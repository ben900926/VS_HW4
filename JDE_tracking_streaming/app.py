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

import torch
from tracker.multitracker import JDETracker
from utils import visualization as vis


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
cfg_dict = parse_model_cfg(opt.cfg)
opt.img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]

logger.setLevel(logging.INFO)
app = Flask(__name__)
if opt.input_video.endswith('.mp4'):
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
elif opt.input_video == '0':
    dataloader = datasets.LoadStreams(opt.input_video, opt.img_size)
else:
    raise NotImplementedError()
tracker = JDETracker(opt, frame_rate = dataloader.frame_rate)


## used for jpeg stream
def yield_frame(opt, dataloader):
    '''
       Processes the video sequence given and provides the output of tracking result (write the results in video file)

       It uses JDE model for getting information about the online targets present.

       Parameters
       ----------
       opt : Namespace
             Contains information passed as commandline arguments.

       dataloader : LoadVideo
                    Instance of LoadVideo class used for fetching the image sequence and associated data.

       data_type : String
                   Type of dataset corresponding(similar) to the given video.

       result_filename : String
                         The name(path) of the file for storing results.

       save_dir : String
                  Path to the folder for storing the frames containing bounding box information (Result frames).

       show_image : bool
                    Option for shhowing individial frames during run-time.

       frame_rate : int
                    Frame-rate of the given video.

       Returns
       -------
       (Returns are not significant here)
       frame_id : int
                  Sequence number of the last sequence
       '''
    global tracker

    timer = Timer()
    frame_id = 0
    for path, img, img0 in dataloader:
        # if frame_id % 20 == 0:
        #     logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1./max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        # save results
        online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                        fps=1. / timer.average_time)
        
        ret, buffer = cv2.imencode('.jpg', online_im)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    try:
        return Response(yield_frame(opt, dataloader), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.info(e)


@app.route('/data', methods=["GET", "POST"])
def mouse_click_pos():
    global tracker

    data = json.loads(request.data)
    # x, y: relative coordinate of the click on video
    x, y = data.get('x'), data.get('y')

    # TODO: do something with the track here
    if x is not None and y is not None:
        tracker.update_display(x, y)
    
    # You can return a response if necessary
    return 'success'


if __name__ == '__main__':
    app.run()