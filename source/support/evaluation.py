import numpy as np

class EvaluationKeys():
  def __init__(self):
    pass

  def knapsack(self, values, weights, capacity):
    """
    Solves the 0/1 knapsack problem using dynamic programming.

    Args:
      values: A list of values for each item.
      weights: A list of weights for each item.
      capacity: The maximum weight capacity of the knapsack.

    Returns:
      A tuple containing the maximum value that can be obtained and
      a list of indices of the selected items.
    """

    num_items = len(values) + 1  # Number of items + 1 (for the 0th item)
    capacity_threshold = capacity + 1  # Capacity threshold + 1 (for the 0th capacity)

    # Adding 0 at the 0th index to match the modified size
    values = np.r_[[0], values]  # Adding 0 at the 0th index (concatenating)
    weights = np.r_[[0], weights]  # Adding 0 at the 0th index (concatenating)

    # Creating and filling the dynamic programming table with zeros
    dp_table = [[0 for _ in range(capacity_threshold)] for _ in range(num_items)]

    # Filling the table using dynamic programming
    for i in range(1, num_items):
      for current_capacity in range(1, capacity_threshold):
        if weights[i] <= current_capacity:
          dp_table[i][current_capacity] = max(values[i] + dp_table[i - 1][current_capacity - weights[i]],
                                               dp_table[i - 1][current_capacity])
        else:
          dp_table[i][current_capacity] = dp_table[i - 1][current_capacity]

    # Backtracking to find the selected items
    selected_items = []
    i = num_items - 1
    current_capacity = capacity_threshold - 1
    while i > 0 and current_capacity > 0:
        if dp_table[i][current_capacity] != dp_table[i - 1][current_capacity]:
            selected_items.append(i - 1)
            current_capacity -= weights[i]
            i -= 1
        else:
            i -= 1

    return dp_table[num_items - 1][capacity_threshold - 1], selected_items




  def eval_metrics(self,y_pred, y_true):
    '''Returns precision, recall and f1-score of given prediction and true value'''
    true_positive = np.sum(y_pred * y_true)
    avoid_zero_denominator = 1e-8
    precision = true_positive / (np.sum(y_pred) + avoid_zero_denominator) # true positive + false positive
    recall = true_positive / (np.sum(y_true) + avoid_zero_denominator) # true positive + false negative
    if precision == 0 and recall == 0:
        f1score = 0
    else:
        f1score = 2 * precision * recall / (precision + recall) # F1score = 2*P*R/(P+R)

    return [precision, recall, f1score]



  def upsample(self, downsampled_array, video_length):
    """
    Upsamples a downsampled array to the original video length.

    Args:
        downsampled_array: The downsampled array.
        video_length: The original video length.

    Returns:
        The upsampled array.
    """
    upsampled_array = np.zeros(video_length)
    upsampling_ratio = video_length // 320
    start_index = (video_length - upsampling_ratio * 320) // 2
    index = 0
    while index < 320:
        upsampled_array[start_index : start_index + upsampling_ratio] = (
            np.ones(upsampling_ratio, dtype=int) * downsampled_array[0][index]
        )
        start_index += upsampling_ratio
        index += 1

    return upsampled_array

  def select_keyshots(self, video_info, predicted_scores):
    """
    Selects key shots from a video using the knapsack algorithm.

    Args:
        video_info: A dictionary containing information about the video.
        predicted_scores: A list of predicted scores for each frame.

    Returns:
        A tuple containing:
            - The upsampled predicted scores.
            - A list of selected key shots.
            - A list of binary labels indicating whether each frame is a key shot.
    """
    if isinstance(video_info['length'], int):
        video_length = video_info['length']
    else:  # This else is likely unnecessary as it should be an int
        video_length = video_info['length'][()]

    change_points = video_info['change_points'][:]  # List of change points (shot boundaries)
    segment_weights = video_info['n_frame_per_seg'][:]  # Number of frames in each segment

    predicted_scores = np.array(predicted_scores)  # Convert predicted scores to a NumPy array
    upsampled_scores = self.upsample(predicted_scores, video_length)  # Upsample scores to match the original video length

    # Calculate the average score for each segment
    segment_scores = np.array([upsampled_scores[cp[0]:cp[1]].mean() for cp in change_points])

    # Apply the knapsack algorithm to select key shots
    _, selected_segments = self.knapsack(segment_scores, segment_weights, int(0.15 * video_length))
    selected_segments = selected_segments[::-1]  # Reverse the order of selected segments

    # Create a binary label array indicating key shots
    key_shot_labels = np.zeros((video_length,))
    for segment_index in selected_segments:
        key_shot_labels[change_points[segment_index][0]:change_points[segment_index][1]] = 1

    return upsampled_scores.tolist(), selected_segments, key_shot_labels.tolist()