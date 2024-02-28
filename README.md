# MusicClassification
Music Classification project with implementation of TensorFlow machine learning.

Model used the GTZAN dataset, with 1000 thirty seconds samples, with 10 genres and 100 music per genre.
The project implements the MFCC representation of the music:
```
def preporcess(dataset_path, num_mfcc=40, n_fft=2048, hop_length=512, num_segment=10):
  data = {"labels": [], "mfcc": []}
  classes = []
  sample_rate = 22050
  samples_per_segment = int(sample_rate*30/num_segment)


  for label_idx, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
    if dirpath == dataset_path:
      continue

    for f in sorted(filenames):
      if not f.endswith('.wav'):
        continue

      file_path = dirpath + "/" + f
      label_name = dirpath.split('/')[-1]
      if label_name not in classes:
        classes.append(label_name)
      print("Track name ", file_path)
      print("LABEL: ", label_idx, label_name)


      try:
        y, sr = librosa.load(file_path, sr = sample_rate)
      except:
        print('ERROR')
        continue
      for n in range(num_segment):
        mfcc = librosa.feature.mfcc(y=y[samples_per_segment*n: samples_per_segment*(n+1)],
                            sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

        mfcc = mfcc.T
        if len(mfcc) == math.ceil(samples_per_segment/hop_length):
          data["mfcc"].append(mfcc.tolist())
          data["labels"].append(label_idx)
  return data, classes
```

And then trains a multple layered Neural Network on said data:
```
layers = tf.keras.layers

cnn_model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation = 'relu', padding='valid', input_shape=input_shape),
    layers.MaxPooling2D(2, padding="same"),

    layers.Conv2D(64, (3, 3), activation='relu', padding='valid'),
    layers.MaxPooling2D(2, padding="same"),

    layers.Conv2D(128, (3, 3), activation = 'relu', padding='valid', input_shape=input_shape),
    layers.MaxPooling2D(2, padding="same"),
    layers.Dropout(0.3),

    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation = 'relu'),
    layers.Dense(10, activation = 'softmax')
])
```

The training took a minute and a half, achieving 88% accuracy on the evaluation. It's a light-weight experimental model, supposed to be fast and compact.

# Usage
To test the model after training, assure you upload your file to the ```/content``` folder and name it ```audio.mp3```, then run the last cells of code to get the prediction.
The 10 selected genres included in the training are: blues
- classical
- country
- disco
- hiphop
- jazz
- metal
- pop
- reggae
- rock
