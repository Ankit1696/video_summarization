import math

import numpy as np

from source.frame_selector.kts import KTS


class ChangePointDetector:
    def __init__(self):
        self.kts = KTS()

    def get_change_points(self, video_features, total_frames, frames_per_second):
        video_duration_seconds = total_frames / frames_per_second
        max_change_points = int(math.ceil(video_duration_seconds / 2.0))
        kernel_matrix = np.dot(video_features, video_features.T)
        change_point_indices, _ = self.kts.cpd_auto(kernel_matrix, max_change_points, 1)
        change_point_indices = np.concatenate(([0], change_point_indices, [total_frames - 1]))

        change_point_segments = []
        for i in range(len(change_point_indices) - 1):
            segment = [change_point_indices[i], change_point_indices[i + 1] - 1]
            if i == len(change_point_indices) - 2:
                segment = [change_point_indices[i], change_point_indices[i + 1]]

            change_point_segments.append(segment)
        change_point_segments = np.array(list(change_point_segments))

        frames_per_segment = []
        for i in range(len(change_point_segments)):
            num_frames_in_segment = change_point_segments[i][1] - change_point_segments[i][0]
            frames_per_segment.append(num_frames_in_segment)
        frames_per_segment = np.array(list(frames_per_segment))

        return change_point_segments, frames_per_segment