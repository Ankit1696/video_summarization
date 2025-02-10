import h5py
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, BatchNormalization, Dropout


tvsumh5datasets = '../../../Downloads/tvsum (2).h5'

class DatasetLoader():
  def __init__(self, dataset_path):
    self.dataset = dataset_path
  def __len__(self):
    return len(self.dataset)
  def __getitem__(self,index_of_vidoes):
    index_of_vidoes = index_of_vidoes +1
    video = self.dataset['video_'+str(index_of_vidoes)]
    feature = video['feature'][:]
    label = video['label'][:]

    feature = np.array(feature)
    label = np.array(label)


    return feature, label, index_of_vidoes

class DatasetBatchCreatorForTraining(Sequence):
    def __init__(self, dataset_path, batch_size, shuffle=False):
        self.dataset = dataset_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
          np.random.shuffle(self.indices)

    def __len__(self):
        length_of_dataset = len(self.dataset)/self.batch_size
        length_of_dataset = np.floor(length_of_dataset)
        length_of_dataset = int(length_of_dataset)
        return length_of_dataset

    def __getitem__(self,index):
        indexes = self.indices[index * self.batch_size : (index+1) * self.batch_size]
        feature = np.empty((self.batch_size,320,1024)) #
        label = np.empty((self.batch_size,320,1)) #

        for i in range(len(indexes)):
          feature[i] = np.array(self.dataset[indexes[i]][0])
          label[i] = np.array(self.dataset[indexes[i]][1]).reshape(-1,1)

        return feature, label


datasetLoader = DatasetLoader(h5py.File(tvsumh5datasets))
X, y = train_test_split(datasetLoader, test_size=0.2)

X_with_batch_size = DatasetBatchCreatorForTraining(X,8)

model_one = Sequential()
model_one.add(LSTM(128, return_sequences=True, input_shape=(320, 1024)))
model_one.add(LSTM(256, return_sequences=True))
# model_one.add(Dense(256, activation='relu'))
# model_one.add(Dense(128, activation='relu'))
model_one.add(TimeDistributed(Dense(1, activation='sigmoid')))

model_one.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
model_one.fit(X_with_batch_size, epochs=10)

model_one.save("model.keras")
