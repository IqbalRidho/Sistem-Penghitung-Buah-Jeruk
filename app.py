from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import time  

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    static_folder='static',
    static_url_path='/static'
)

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'results')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO("yolov8n-seg.pt")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))
    original = image.copy()

    results = model(image)

    mask_total = np.zeros(image.shape[:2], dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_img = np.zeros_like(gray)

    count = 0

    for r in results:
        if r.masks is None:
            continue

        for mask in r.masks.data:
            mask = (mask.cpu().numpy() * 255).astype(np.uint8)
            area = cv2.countNonZero(mask)

            if area > 500:
                count += 1
                mask_total = cv2.bitwise_or(mask_total, mask)

                masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
                edges = cv2.Canny(masked_gray, 100, 200)
                edge_img = cv2.bitwise_or(edge_img, edges)

    result = original.copy()
    result[mask_total > 0] = [0, 255, 0]

    paths = {
        "original": os.path.join(RESULT_FOLDER, "original.jpg"),
        "mask": os.path.join(RESULT_FOLDER, "mask.jpg"),
        "edges": os.path.join(RESULT_FOLDER, "edges.jpg"),
        "result": os.path.join(RESULT_FOLDER, "result.jpg"),
    }

    cv2.imwrite(paths["original"], original)
    cv2.imwrite(paths["mask"], mask_total)
    cv2.imwrite(paths["edges"], edge_img)
    cv2.imwrite(paths["result"], result)

    return count, paths


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify(success=False, error="File tidak ditemukan")

    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify(success=False, error="Format file tidak valid")

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    count, images = process_image(path)
    

    timestamp = int(time.time() * 1000)

    return jsonify(
        success=True,
        count=count,
        images={k: f"/results/{os.path.basename(v)}?t={timestamp}" for k, v in images.items()}
    )


@app.route('/results/<filename>')
def results(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename))


if __name__ == '__main__':
    app.run(debug=True)
