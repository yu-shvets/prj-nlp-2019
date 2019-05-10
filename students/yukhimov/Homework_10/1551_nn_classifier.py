from keras.models import Sequential
from keras.layers import Dense
import json
import numpy
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

encoder = LabelEncoder()

with open('1551_features.txt') as json_file:
    data = json.load(json_file)

train_features, train_labels = numpy.array(data['train_features']), \
                               to_categorical(encoder.fit_transform(data['train_labels']))
test_features, test_labels = numpy.array(data['test_features']), \
                             to_categorical(encoder.transform(data['test_labels']))

model = Sequential()
model.add(Dense(32, input_dim=300, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(len(set(data['train_labels'])), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_features, train_labels, epochs=150, batch_size=10)

score = model.evaluate(test_features, test_labels)

print("Accuracy: {}".format(score[1]))

# Accuracy: 0.49044069994162937
