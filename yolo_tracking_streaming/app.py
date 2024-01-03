from flask import Flask, render_template, Response, request
import json
import cv2
import argparse

import sys
sys.path.insert(0, './yolov5')
from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import time
import torch
import torch.backends.cudnn as cudnn
from track import draw_boxes, xyxy_to_xywh


app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5x_50.pt', help='model.pt path') # yolov5s.pt
parser.add_argument('--deep_sort_weights', type=str, 
        default='deep_sort_pytorch/deep_sort/deep/checkpoint/original_ckpt.t7', help='ckpt.t7 path')
# file/folder, 0 for webcam
parser.add_argument('--source', type=str, default='0', help='source')
parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
# parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
# parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
# parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
# class 0 is person, 1 is bycicle, 2 is car... 79 is oven
parser.add_argument('--classes', nargs='+', default=[0], type=int, help='filter by class')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--evaluate', action='store_true', help='augmented inference')
parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
## to 
# parser.add_argument('--img_stream', action='store_true', help='to image flow that can be used for streaming')
##
args = parser.parse_args()

# get deepsort tracker (global variable)
cfg = get_config()
cfg.merge_from_file(args.config_deepsort)
attempt_download(args.deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)


def generate_frames(opt):
    global deepsort

    opt.img_size = check_img_size(opt.img_size)
    source, yolo_weights, imgsz = opt.source, opt.yolo_weights, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    
    # Initialize
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                im0 = im0s[i].copy()
            else:
                im0 = im0s

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                xywh_bboxs = []
                confs = []

                # Adapt detections to input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)

                # pass detections to tracking model... (Hungarian)
                outputs = deepsort.update(xywhs, confss, im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1] # [1, 2, 3...] (list of shown id)

                    draw_boxes(im0, bbox_xyxy, identities)
            else:
                deepsort.increment_ages()

            ret, buffer = cv2.imencode('.jpg', im0)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(args), mimetype='multipart/x-mixed-replace; boundary=frame')


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