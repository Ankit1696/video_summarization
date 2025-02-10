from flask import Flask, render_template, request, send_file, url_for, redirect
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from source.datageneration.datageneration import VideoProcessing, EvaluationKeys
from tensorflow.keras.models import load_model
import time
import uuid
from threading import Lock
from flask import jsonify
from flask_cors import CORS

# Add these after model loading
progress = {}
progress_lock = Lock()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SUMMARY_FOLDER'] = 'static/summaries'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

# Load model once
model = load_model('model.keras', compile=False)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_video(input_path, output_path, progress_callback=None):
    # Initialize VideoProcessing with progress callback
    video_processing = VideoProcessing(input_path, progress_callback=progress_callback)
    video_info = video_processing.get_processed_data()

    video_feature = np.array(video_info['feature'][:])
    predicted_score = model.predict(video_feature.reshape(-1, 320, 1024))

    evaluation_keys = EvaluationKeys()
    _, _, predicted_summary = evaluation_keys.select_keyshots(video_info, predicted_score)

    video = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # For H.264 encoding (if supported)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        raise ValueError("Video has no frames")

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_index = 0
    success, frame = video.read()
    while success:
        if predicted_summary[frame_index] == 1:
            out.write(frame)
        frame_index += 1
        if progress_callback and frame_index % 10 == 0:
            # Correct progress calculation
            progress_percent = int((frame_index / total_frames) * 100)
            progress_callback(progress_percent)
        success, frame = video.read()

    video.release()
    out.release()
    # Ensure final progress is 100%
    if progress_callback:
        progress_callback(100)



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file selected')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)

            # Generate unique output filename
            summary_filename = f"summary_{filename}"
            output_path = os.path.join(app.config['SUMMARY_FOLDER'], summary_filename)

            # Process video
            try:
                process_video(input_path, output_path)
                return render_template('result.html',
                                       original_video=filename,
                                       summary_video=summary_filename)
            except Exception as e:
                return render_template('index.html', error=str(e))

    return render_template('index.html')


@app.route('/result')
def result():
    original_video = request.args.get('original')
    summary_video = request.args.get('summary')
    if not original_video or not summary_video:
        return redirect(url_for('index'))
    return render_template('result.html',
                           original_video=original_video,
                           summary_video=summary_video)


# @app.route('/download/<filename>')
# def download(filename):
#     return send_file(os.path.join(app.config['SUMMARY_FOLDER'], filename),
#                      as_attachment=True)

@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    return send_file(os.path.join(app.config['SUMMARY_FOLDER'], filename), as_attachment=True)


@app.route('/progress/<task_id>')
def get_progress(task_id):
    with progress_lock:
        return jsonify(progress.get(task_id, 0))


@app.route('/process', methods=['POST'])
def process_task():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        summary_filename = f"summary_{filename}"
        output_path = os.path.join(app.config['SUMMARY_FOLDER'], summary_filename)

        task_id = str(uuid.uuid4())

        def update_progress(p):
                with progress_lock:
                    progress[task_id] = p

        from threading import Thread
        thread = Thread(target=process_video, args=(input_path, output_path),
                        kwargs={'progress_callback': update_progress})
        thread.start()

        return jsonify({'task_id': task_id, 'summary_filename': summary_filename, 'original_filename': filename})

    return jsonify({'error': 'Invalid file type'}), 400



if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['SUMMARY_FOLDER'], exist_ok=True)
    app.run(debug=False, port=5001)