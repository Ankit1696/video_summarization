import os

import cv2
import numpy as np
import torch
from tqdm import tqdm
from source.dataloader.rawdata import TvSum50MatlabData, TvSum50VideoData
from source.summary.summmary_generation import SummaryGeneration
from source.support.cps import ChangePointDetector
from source.support.evaluation import EvaluationKeys
import h5py

from source.support.feature_extraction import extract_features


class DataGeneration:
    def __init__(self):
        self.summary_generation = SummaryGeneration()
        self.evaluation_keys = EvaluationKeys()
        self.change_point_detector=ChangePointDetector()
        self.tv_sum_data = TvSum50MatlabData('../../../Desktop/Dissertation/DATA/tvsum50_ver_1_1/ydata-tvsum50-v1_1/matlab/ydata-tvsum50.mat')
        self.tv_sum_video_path = TvSum50VideoData('../../../Desktop/Dissertation/DATA/tvsum50_ver_1_1/ydata-tvsum50-v1_1/video')

    def create_h5_dataset_training(self):
        video_name_mapping = {}
        for i in range(len(self.tv_sum_data.data['tvsum50']['video'])):
            video_name_mapping[i+1] = self.tv_sum_data.data['tvsum50']['video'][i]

        with h5py.File("tvsum.h5", "w") as f:
            for video_index in tqdm(range(len(self.tv_sum_data.data['tvsum50']['video'])), desc="Video", ncols=80, leave=False):
                video_path = os.path.join(self.tv_sum_video_path.video_path, video_name_mapping[video_index+1]+'.mp4')
                video_group_name = 'video_' + str(video_index + 1)
                video_group = f.create_group(video_group_name)

                video = cv2.VideoCapture(video_path)
                tqdm.write('Processing video ' + str(video_index+1))

                video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                frames_per_second = video.get(cv2.CAP_PROP_FPS)
                frame_sampling_ratio = video_frame_count // 320
                frame_offset = (video_frame_count - frame_sampling_ratio * 320) // 2
                all_features = []
                training_features = []
                selected_frame_indices = []
                frame_index = 0
                picked_frame_count = 0
                success, frame = video.read()
                while success:
                    frame_feature = extract_features(frame)

                    if (frame_index + 1) > frame_offset and picked_frame_count < 320:
                        if (frame_index + 1 - frame_offset) % frame_sampling_ratio == 0:
                            selected_frame_indices.append(frame_index)
                            picked_frame_count += 1
                            training_features.append(frame_feature)

                    all_features.append(frame_feature)
                    frame_index += 1

                    success, frame = video.read()

                video.release()

                all_features = torch.stack(all_features).numpy()
                training_features = torch.stack(training_features).numpy()

                change_points, frames_per_segment = self.change_point_detector.get_change_points(all_features, video_frame_count, frames_per_second)

                user_summary = self.summary_generation.generate_user_summary_tvsum(
                    video_index, video_frame_count, change_points, frames_per_segment)

                oracle_summary = self.summary_generation.generate_user_summary_tvsum(user_summary)
                frame_labels = [oracle_summary[picked_frame] for picked_frame in selected_frame_indices]

                video_group['feature'] = training_features
                video_group['label'] = frame_labels
                video_group['length'] = video_frame_count
                video_group['change_points'] = change_points
                video_group['n_frame_per_seg'] = frames_per_segment
                video_group['picks'] = np.array(list(selected_frame_indices))
                video_group['user_summary'] = user_summary


class VideoProcessing:
    def __init__(self, video_path, progress_callback=None):
        self.change_point_detector = ChangePointDetector()
        self.video_file_path = video_path
        self.progress_callback = progress_callback

    def get_processed_data(self):
        video_index = 0
        video_path = self.video_file_path

        video = cv2.VideoCapture(video_path)
        tqdm.write('Processing video... ' + str(video_index + 1))

        video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        frame_sampling_ratio = video_frame_count // 320
        frame_offset = (video_frame_count - frame_sampling_ratio * 320) // 2
        all_frames_features = []
        training_features = []
        selected_frame_indices = []
        frame_index = 0
        picked_frame_count = 0
        success, frame = video.read()
        while success:
            frame_feature = extract_features(frame)

            if (frame_index + 1) > frame_offset and picked_frame_count < 320:
                if (frame_index + 1 - frame_offset) % frame_sampling_ratio == 0:
                    selected_frame_indices.append(frame_index)
                    picked_frame_count += 1
                    training_features.append(frame_feature)

            all_frames_features.append(frame_feature)
            frame_index += 1

            # Update progress during feature extraction
            if self.progress_callback and frame_index % 10 == 0:
                progress_percent = int((frame_index / video_frame_count) * 50)  # 50% for feature extraction
                self.progress_callback(progress_percent)

            success, frame = video.read()

        video.release()

        all_frames_features = torch.stack(all_frames_features).numpy()
        training_features = torch.stack(training_features).numpy()


        change_points, frames_per_segment = self.change_point_detector.get_change_points(all_frames_features,
                                                                                         video_frame_count,
                                                                                         frames_per_second)

        # Update progress after change point detection
        if self.progress_callback:
            self.progress_callback(75)  # 75% for change point detection

        video_info = {
            'feature': training_features,
            'length': video_frame_count,
            'change_points': change_points,
            'n_frame_per_seg': frames_per_segment,
            'picks': np.array(list(selected_frame_indices))
        }

        return video_info