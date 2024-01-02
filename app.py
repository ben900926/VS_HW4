from flask import Flask, render_template, Response, request
import json
import cv2


app = Flask(__name__)
camera = cv2.VideoCapture(0)  # Use the default camera


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/data', methods=["GET", "POST"])
def mouse_click_pos():
    data = json.loads(request.data)
    # x, y: relative coordinate of the click on video
    x, y = data.get('x'), data.get('y')

    # TODO: do something with the track here
    print("x:", x, "y:", y)
    
    # You can return a response if necessary
    return 'success'


if __name__ == '__main__':
    app.run()
