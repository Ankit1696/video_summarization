import mat73


class TvSum50MatlabData:
    def __init__(self,path):
        self.path = path
        self.data = mat73.loadmat(path)


class TvSum50VideoData:
    def __init__(self,video_path):
        self.video_path = video_path



