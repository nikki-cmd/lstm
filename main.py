import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
import keras.utils as ku 
from keras import ops


# set seeds for reproducability
import tensorflow as tf
from numpy.random import seed
tf.random.set_seed(2)
seed(1)

import pandas as pd
import numpy as np
import string, os 

import warnings

tokenizer = Tokenizer()

def clean_text(txt):
	txt = "".join(v for v in txt if v not in string.punctuation).lower()
	txt = txt.encode("utf8").decode("ascii", 'ignore')
	return txt


def get_sequence_of_tokens(corpus):
	
	tokenizer.fit_on_texts(corpus)
	total_words = len(tokenizer.word_index) + 1

	input_sequences = []
	for line in corpus:
		token_list = tokenizer.texts_to_sequences([line])[0]
		for i in range(1, len(token_list)):
			n_gram_sequence = token_list[:i + 1]
			input_sequences.append(n_gram_sequence)
	return input_sequences, total_words

def generate_padded_sequences(input_sequences):
	max_sequence_len = max([len(x) for x in input_sequences])
	input_sequences = np.array(pad_sequences(input_sequences, maxlen = max_sequence_len, padding='pre'))
	predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
	label = ku.to_categorical(label, num_classes=total_words)
	return predictors, label, max_sequence_len

def create_model(max_sequence_len, total_words):
	input_len = max_sequence_len - 1
	model = Sequential()

	model.add(Embedding(total_words, 10, input_length=input_len))

	model.add(LSTM(100))
	model.add(Dropout(0.1))

	model.add(Dense(total_words, activation="softmax"))

	model.compile(loss='categorical_crossentropy', optimizer='adam')

	return model


def generate_text(seed_text, next_words, model, max_sequence_len):
	for _ in range(next_words):
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
		predicted = model.predict(token_list)
		predicted = np.argmax(predicted, axis=1)

		output_word = ""
		for word,index in tokenizer.word_index.items():
			if index == predicted:
				output_word = word
				break
		seed_text += " " + output_word
	return seed_text.title()

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

curr_dir = "dataset/"
all_headlines = []

for filename in os.listdir(curr_dir):
	if 'Articles' in filename:
		article_df = pd.read_csv(curr_dir + filename)
		all_headlines.extend(list(article_df.headline.values))
		break

all_headlines = [h for h in all_headlines if h != "Unknown"]
print(len(all_headlines))

corpus = [clean_text(x) for x in all_headlines]
print(corpus[:10])

inp_sequences, total_words = get_sequence_of_tokens(corpus)
print(inp_sequences[:10])

predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)

model =create_model(max_sequence_len, total_words)
model.summary()
 
to_train = False
if to_train:
    epochs = 1000
    verbose = 5
    model.fit(predictors, label, epochs=epochs, verbose=verbose)
    model.save('models/model_test_1000epochs.keras')

model = load_model("models/model_test_1000epochs.keras")

print (generate_text("united states", 20, model, max_sequence_len))
print (generate_text("preident trump", 10, model, max_sequence_len))
print (generate_text("donald trump", 20, model, max_sequence_len))
print (generate_text("india and china", 10, model, max_sequence_len))
print (generate_text("new york", 20, model, max_sequence_len))
print (generate_text("science and technology", 10, model, max_sequence_len))

request = ""
while request != "stop":
    print(generate_text(input(), 20, model, max_sequence_len))