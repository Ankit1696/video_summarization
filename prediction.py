# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return render_template('index.html')

# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application
    # on the local development server.
    app.run()






import cv2
import numpy as np
from datageneration.datageneration import VideoProcessing
from tensorflow.keras.models import load_model
from datageneration.datageneration import  EvaluationKeys

model = load_model('../training/model.keras', compile=False)
model.summary()

# model.predict(np.array([[1,2,3]]))



video_file = '../../../Desktop/Dissertation/DATA/tvsum50_ver_1_1/ydata-tvsum50-v1_1/video/_xMr-HKMfVA.mp4'
video_processing = VideoProcessing(video_file)
video_info = video_processing.get_processed_data()


video_feature = video_info['feature'][:]
video_feature = np.array(video_feature)


predicted_score = model.predict(video_feature.reshape(-1,320,1024))
evaluation_keys = EvaluationKeys()
predicted_scores, selected_frames, predicted_summary =evaluation_keys.select_keyshots(video_info, predicted_score)


video = cv2.VideoCapture(video_file)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = video.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("summary.mp4", fourcc, fps, (int(video.get(3)), int(video.get(4))))

frame_index = 0
success, frame = video.read()
while success:
    if predicted_summary[frame_index] == 1:
        out.write(frame)
    frame_index += 1
    success, frame = video.read()

video.release()
out.release()

