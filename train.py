import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.optimizers import Adam
import numpy as np

# Load the data
train_data = pd.read_csv('./cnn_dailymail/train.csv')
test_data = pd.read_csv('./cnn_dailymail/test.csv')

# Assuming you have a separate validation dataset or split
validation_data = pd.read_csv('./cnn_dailymail/validation.csv')

# Preprocess the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['article'])
tokenizer.fit_on_texts(train_data['highlights'])
tokenizer.fit_on_texts(validation_data['article'])
tokenizer.fit_on_texts(validation_data['highlights'])

train_article_sequences = tokenizer.texts_to_sequences(train_data['article'])
train_highlight_sequences = tokenizer.texts_to_sequences(train_data['highlights'])
validation_article_sequences = tokenizer.texts_to_sequences(validation_data['article'])
validation_highlight_sequences = tokenizer.texts_to_sequences(validation_data['highlights'])

max_len_article = 200  # Set to an appropriate value
max_len_highlight = 200  # Set to an appropriate value

train_article_sequences_np = pad_sequences(train_article_sequences, maxlen=max_len_article, padding='post')
train_highlight_sequences_np = pad_sequences(train_highlight_sequences, maxlen=max_len_highlight, padding='post')
validation_article_sequences_np = pad_sequences(validation_article_sequences, maxlen=max_len_article, padding='post')
validation_highlight_sequences_np = pad_sequences(validation_highlight_sequences, maxlen=max_len_highlight, padding='post')


# Define the model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=20, input_length=max_len_article))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Train the model
batch_size = 2
epochs = 3

history = model.fit(
    train_article_sequences_np,
    train_highlight_sequences_np,
    epochs=epochs,
    validation_data=(validation_article_sequences_np, validation_highlight_sequences_np),
    batch_size=batch_size
)

# Save the final model
model.save('text_summarization_model.h5')
