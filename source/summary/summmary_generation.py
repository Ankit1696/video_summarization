from source.support.evaluation import EvaluationKeys
import numpy as np
from source.dataloader.rawdata import TvSum50MatlabData


class SummaryGeneration:
    def __init__(self):
        self.evaluation_keys = EvaluationKeys()
        self.tv_sum_data = TvSum50MatlabData('../../../Desktop/Dissertation/DATA/tvsum50_ver_1_1/ydata-tvsum50-v1_1/matlab/ydata-tvsum50.mat')

    def generate_user_summary_tvsum(self, video_index, video_length, change_points, segment_weights):
        num_users = len(self.tv_sum_data.data['tvsum50']['user_anno'][video_index][0])
        user_summary = np.zeros((num_users, video_length))
        user_annotations = np.zeros((num_users, video_length))
        for i in range(video_length):
            for j in range(num_users):
                user_annotations[j][i] = self.tv_sum_data['tvsum50']['user_anno'][video_index][i][j]
        for i in range(num_users):
            ground_truth_values = np.array([user_annotations[cp[0]:cp[1]].mean() for cp in change_points])
            _, selected_segments = self.evaluation_keys.knapsack(ground_truth_values, segment_weights, int(0.15 * video_length))

            selected_segments = selected_segments[::-1]
            key_shot_labels = np.zeros((video_length,))
            for j in selected_segments:
                key_shot_labels[change_points[j][0]:change_points[j][1]] = 1
            user_summary[i, ] = key_shot_labels
        return user_summary

    def get_oracle_summary(self, user_summary):
        num_users, num_frames = user_summary.shape
        oracle_summary = np.zeros(num_frames)
        overlap_array = np.zeros(num_users)
        oracle_sum = 0
        true_summary_lengths = user_summary.sum(axis=1)
        priority_indices = np.argsort(-user_summary.sum(axis=0))
        best_f_score = 0
        for index in priority_indices:
            oracle_sum += 1
            for user_index in range(num_users):
                overlap_array[user_index] += user_summary[user_index][index]
            current_f_score = self.evaluation_keys.eval_metrics(overlap_array, true_summary_lengths, oracle_sum)

            if current_f_score > best_f_score:
                best_f_score = current_f_score
                oracle_summary[index] = 1
            else:
                break
        print(f'Overlap: {overlap_array}')
        print(f'True summary n_key: {true_summary_lengths}')
        print(f'Oracle summary n_key: {oracle_sum}')
        print(f'Final F-score: {best_f_score}')
        return oracle_summary