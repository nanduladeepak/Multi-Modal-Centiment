import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Load the WAV file

wav_file_path = 'path/to/your/audio.wav'
audio, sr = librosa.load(wav_file_path, sr=None)

# Step 2: Divide the audio into segments

segment_length = 3

num_segments = len(audio) // (sr * segment_length)
segments = np.array_split(audio[:num_segments * sr * segment_length], num_segments)

# Step 3: Compute acoustic features

def compute_mfcc(audio, sr):
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
    return mfcc

features = [compute_mfcc(segment, sr) for segment in segments]

# Step 4: Apply Contrastive Predictive Coding

class ASLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ASLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.linear(output[:, -1, :])
        return output

input_dim = features[0].shape[1]
hidden_dim = 64
output_dim = 256

model = ASLSTM(input_dim, hidden_dim, output_dim)

criterion = nn.ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

features_tensor = torch.tensor(features, dtype=torch.float32)

# Step 5: Training the A-sLSTM model

num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    positive_pairs = features_tensor[:-1]
    negative_pairs = features_tensor[np.random.choice(num_segments - 1, len(positive_pairs))]

    all_pairs = torch.cat((positive_pairs, negative_pairs), dim=0)

    shuffled_indices = torch.randperm(len(all_pairs))
    shuffled_pairs = all_pairs[shuffled_indices]

    predictions = model(all_pairs)
    shuffled_predictions = model(shuffled_pairs)

    loss = criterion(predictions, shuffled_predictions)

    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

# Step 6: Fine-tuning or downstream tasks

model.eval()
with torch.no_grad():
    representations = model(features_tensor)

# Use the learned representations for your specific downstream task

# For example, you can perform speech recognition using the representations as input to another model.




#===========================================================================================================================================================================================

import tensorflow as tf
import numpy as np
import librosa

# Step 1: Load the WAV file

wav_file_path = 'path/to/your/audio.wav'
audio, sr = librosa.load(wav_file_path, sr=None)

# Step 2: Divide the audio into segments

segment_length = 3

num_segments = len(audio) // (sr * segment_length)
segments = np.array_split(audio[:num_segments * sr * segment_length], num_segments)

# Step 3: Compute acoustic features

def compute_mfcc(audio, sr):
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
    return mfcc

features = [compute_mfcc(segment, sr) for segment in segments]

# Step 4: Apply Contrastive Predictive Coding

# Set up the A-sLSTM model using TensorFlow

class ASLSTM(tf.keras.Model):
    def __init__(self, hidden_dim, output_dim):
        super(ASLSTM, self).__init__()
        self.lstm = tf.keras.layers.LSTM(hidden_dim, return_sequences=True)
        self.flatten = tf.keras.layers.Flatten()
        self.linear = tf.keras.layers.Dense(output_dim)

    def call(self, inputs):
        output = self.lstm(inputs)
        output = self.flatten(output[:, -1, :])
        output = self.linear(output)
        return output

input_dim = features[0].shape[1]
hidden_dim = 64
output_dim = 256

model = ASLSTM(hidden_dim, output_dim)

# Set up the contrastive loss function

def contrastive_loss(y_true, y_pred):
    margin = 1.0
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean((1 - y_true) * square_pred + y_true * margin_square)

# Set up the optimizer

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Convert the features to TensorFlow tensors

features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)

# Step 5: Training the A-sLSTM model

num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    indices = np.arange(num_segments - 1)
    np.random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]

        positive_pairs = features_tensor[:-1][batch_indices]
        negative_pairs = tf.gather(features_tensor, np.random.choice(indices, len(batch_indices)), axis=0)

        all_pairs = tf.concat([positive_pairs, negative_pairs], axis=0)

        with tf.GradientTape() as tape:
            predictions = model(all_pairs, training=True)
            shuffled_predictions = tf.random.shuffle(predictions)

            loss = contrastive_loss(tf.ones_like(predictions[:, 0]), predictions - shuffled_predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.numpy()}")

# Step 6: Fine-tuning or downstream tasks

model.evaluate(features_tensor)  # Use the model for downstream tasks


