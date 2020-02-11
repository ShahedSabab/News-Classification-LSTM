import csv 
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

def file_read(name):
    labels = []
    articles = []
    STOPWORDS = set(stopwords.words('english'))
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            labels.append(row[0])
            article = row[1]
            for word in STOPWORDS:
                token = ' ' + word + ' '
                article = article.replace(token, ' ')
                article = article.replace('  ' , ' ')
            articles.append(article)
    return labels, articles

def check_article(word_index, document):
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    return ' '.join([reverse_word_index.get(text, '?') for text in document])

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        # aplying activation
        tf.keras.layers.Dense(embedding_dim, activation='relu',kernel_regularizer=regularizers.l1(.01)),
        #applying softmax for converting output layers to probability distribution 
        tf.keras.layers.Dense(6, activation = 'softmax')
        ])
    print(model.summary())
    return model

def plot_graphs(f, history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

file_name = "bbc-text.csv"


vocab_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
# out of vocabulary token
oov_tok = '<OOV>'
test_portion = 0.20
padding_type = 'post'
num_epoch = 20

labels, articles = file_read(file_name)

# Convert the dataset into train and test datasets
train_articles, test_articles, train_labels, test_labels = train_test_split(articles, labels, test_size=test_portion, random_state=41)

#vectorize the texts 
tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_articles)
test_sequences = tokenizer.texts_to_sequences(test_articles)

#label
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_sequences = np.array(label_tokenizer.texts_to_sequences(train_labels))
test_label_sequences = np.array(label_tokenizer.texts_to_sequences(test_labels))

#padding 0's to make all the samples as same size
train_padded = pad_sequences(train_sequences, maxlen = max_length, padding=padding_type, truncating = trunc_type)
test_padded = pad_sequences(test_sequences, maxlen = max_length, padding=padding_type, truncating = trunc_type)

#decode document using word_index as key
document = train_padded[0]
temp = check_article(word_index, document)


#model save
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

#define model
model = create_model()


model.compile(loss='sparse_categorical_crossentropy', optimizer ='adam', metrics = ['accuracy'])


history = model.fit(train_padded, 
                    training_label_sequences, 
                    epochs = num_epoch, 
                    validation_data = (test_padded, test_label_sequences),
                    callbacks=[cp_callback],
                    verbose=2)

f1 = plt.figure()
plot_graphs(f1,history, "accuracy")
f2 = plt.figure()
plot_graphs(f2,history, "loss")

# Save the entire model as a SavedModel.
#my_model_path = os.path.dirname('saved_model/my_model')
#model.save(my_model_path) 

# =============================================================================
# #load checkpoint (load weights)
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# 
# # Load the previously saved weights
# model.load_weights(latest)
# 
# =============================================================================
# =============================================================================
# Loading model
# my_model_path = os.path.dirname('saved_model/my_model')
# new_model = tf.keras.models.load_model(my_model_path)
# 
# # Check its architecture
# new_model.summary()
# =============================================================================

# Evaluate the model
loss, acc = model.evaluate(test_padded,  test_label_sequences, verbose=2)
print("Test, accuracy: {:5.2f}%".format(100*acc))





