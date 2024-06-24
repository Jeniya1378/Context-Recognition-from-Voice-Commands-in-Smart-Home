import sys 
import pandas as pd
import numpy as np
import string
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection  import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping
from keras import layers
from tensorflow.keras import Sequential, Model
from keras.layers import Embedding, SpatialDropout1D, Conv1D, BatchNormalization, GlobalMaxPool1D, Dropout, Dense, SimpleRNN, LSTM, Bidirectional, Input, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate 
from tensorflow.keras.metrics import AUC
import tensorflow_addons as tfa
from gensim.models.word2vec import Word2Vec
# import matplotlib.pyplot as plt


# Dataset selection
dataset_selection = sys.argv[1]

df1=pd.read_excel(r"datasets\multi label dataset of small talk in smart home (2).xlsx")
df1=df1[['Voice', 'Context']]

df2=pd.read_excel(r"datasets\dataset_refined_context.xlsx")
df2=df2[['Voice', 'Context']]

if dataset_selection == "1":
    df=df1

elif dataset_selection == "2":
    df=df2

elif dataset_selection == "3" or dataset_selection == "4":
    frames = [df1, df2]  
    df = pd.concat(frames)

print("Dataset:")
print(df.head())

#Cleaning
totalContentCleaned = []
punctDict = {}
for punct in string.punctuation:
    punctDict[punct] = None
transString = str.maketrans(punctDict)
for sen in df['Voice']:
    p = sen.translate(transString)
    totalContentCleaned.append(p)
df['clean_text'] = totalContentCleaned


#Literal eval
df['Context'] = df['Context'].apply(lambda x: ast.literal_eval(x))

#Label Binarization
multilabel = MultiLabelBinarizer()
y = multilabel.fit_transform(df['Context'])
numOfClasses=len(multilabel.classes_)
y=pd.DataFrame(y, columns=multilabel.classes_)
Y=y[[x for x in multilabel.classes_]]
df=df.reset_index()
Y=Y.reset_index()
df = pd.merge(Y,df,on='index')

# train test split
train, test = train_test_split(df, train_size=0.7, test_size=0.3, random_state=42)

X_train = train["clean_text"]
X_test  = test["clean_text"]
Y=y[[x for x in multilabel.classes_]]
y_train = train[[x for x in multilabel.classes_]]
y_test  = test[[x for x in multilabel.classes_]]

#embedding parameters
num_words = 20000 
max_features = 10000 
max_len = 200 
embedding_dims = 128 
num_epochs = 10 
val_split = 0.1
batch_size2 = 256 
sequence_length = 250


#Tokenization
tokenizer = Tokenizer(num_words)
tokenizer.fit_on_texts(list(X_train))

#Convert tokenized sentence to sequnces
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
 
# padding the sequences
X_train = pad_sequences(X_train, max_len)
X_test  = pad_sequences(X_test,  max_len)

X_tra, X_val, y_tra, y_val = train_test_split(X_train, y_train, train_size =0.7, random_state=233)

#Early stopping to stop overfitting
early = EarlyStopping(monitor="loss", mode="auto", patience=4)


#Assign weights to contexts
class_frequencies = np.sum(y_train, axis=0)
total_samples = len(y_train)
class_weights = total_samples / (len(class_frequencies) * class_frequencies)
class_weights_dict = {idx: weight for idx, weight in enumerate(class_weights)}


#word2vec
mes = []
for i in df['clean_text']:
    mes.append(i.split())
word2vec_model = Word2Vec(mes, vector_size=100, window=3, min_count=1, workers=16)


# Prepare the embedding matrix vectors in order to feed/pass the neural network
embedding_matrix = np.zeros((len(tokenizer.word_index)+1, 100))

for word, i in tokenizer.word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]


#Convolutional Neural Network (CNN)
print("\n\n-----------------------CNN---------------------\n\n")
CNN_w2v_model = Sequential([
    Embedding(input_dim =embedding_matrix.shape[0], input_length=max_len, output_dim=embedding_matrix.shape[1],weights=[embedding_matrix], trainable=False),
    SpatialDropout1D(0.5),
    Conv1D(filters=100, kernel_size=4, padding='same', activation='relu'),
    BatchNormalization(),
    GlobalMaxPool1D(),
    Dropout(0.5),
    Dense(50, activation = 'relu'),
    Dense(numOfClasses, activation = 'sigmoid')
])
CNN_w2v_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=[tfa.metrics.F1Score(num_classes=numOfClasses,average='weighted'),'binary_accuracy','accuracy', AUC()])
CNN_w2v_model.summary()

if dataset_selection == "4":
    CNN_w2v_model_fit = CNN_w2v_model.fit(X_tra, y_tra, class_weight=class_weights_dict,batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early])
else:
    CNN_w2v_model_fit = CNN_w2v_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early])


# # Plot training & validation accuracy values
# plt.plot(CNN_w2v_model_fit.history['binary_accuracy'])
# plt.plot(CNN_w2v_model_fit.history['val_binary_accuracy'])
# plt.title('CNN-word2vec Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(CNN_w2v_model_fit.history['loss'])
# plt.plot(CNN_w2v_model_fit.history['val_loss'])
# plt.title('CNN-word2vec Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Training Loss', 'Validation Loss'], loc='lower right')
# plt.show()

#Recurrent Neural Network (RNN)
print("\n\n-----------------------RNN---------------------\n\n")
RNN_w2v_model = Sequential([
    Embedding(input_dim =embedding_matrix.shape[0], input_length=max_len, output_dim=embedding_matrix.shape[1],weights=[embedding_matrix], trainable=False),
    SpatialDropout1D(0.5),
    SimpleRNN(25, return_sequences=True),
    BatchNormalization(),
    Dropout(0.5),
    GlobalMaxPool1D(),
    Dense(50, activation = 'relu'),
    Dense(numOfClasses, activation = 'sigmoid')
])
RNN_w2v_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=[tfa.metrics.F1Score(num_classes=numOfClasses,average='weighted'),'binary_accuracy','accuracy', AUC()])
RNN_w2v_model.summary()

if dataset_selection == "4":
    RNN_w2v_model_fit = RNN_w2v_model.fit(X_tra, y_tra, class_weight=class_weights_dict,batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early])
else:
    RNN_w2v_model_fit = RNN_w2v_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early])

# # Plot training & validation accuracy values
# plt.plot(RNN_w2v_model_fit.history['binary_accuracy'])
# plt.plot(RNN_w2v_model_fit.history['val_binary_accuracy'])
# plt.title('RNN-word2vec Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(RNN_w2v_model_fit.history['loss'])
# plt.plot(RNN_w2v_model_fit.history['val_loss'])
# plt.title('RNN-word2vec Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Training Loss', 'Validation Loss'], loc='lower right')
# plt.show()


#Long Short Term Memory (LSTM)
print("\n\n-----------------------LSTM---------------------\n\n")
LSTM_w2v_model = Sequential([
    Embedding(input_dim =embedding_matrix.shape[0], input_length=max_len, output_dim=embedding_matrix.shape[1],weights=[embedding_matrix], trainable=False),
    SpatialDropout1D(0.5),
    LSTM(25, return_sequences=True),
    BatchNormalization(),
    Dropout(0.5),
    GlobalMaxPool1D(),
    Dense(50, activation = 'relu'),
    Dense(numOfClasses, activation = 'sigmoid')
])
LSTM_w2v_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=[tfa.metrics.F1Score(num_classes=numOfClasses,average='weighted'),'binary_accuracy','accuracy', AUC()])
LSTM_w2v_model.summary()

if dataset_selection == "4":
    LSTM_w2v_model_fit = LSTM_w2v_model.fit(X_tra, y_tra,class_weight=class_weights_dict, batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early])
else:
    LSTM_w2v_model_fit = LSTM_w2v_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early])

# # Plot training & validation accuracy values
# plt.plot(LSTM_w2v_model_fit.history['binary_accuracy'])
# plt.plot(LSTM_w2v_model_fit.history['val_binary_accuracy'])
# plt.title('LSTM-word2vec Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(LSTM_w2v_model_fit.history['loss'])
# plt.plot(LSTM_w2v_model_fit.history['val_loss'])
# plt.title('LSTM-word2vec Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Training Loss', 'Validation Loss'], loc='lower right')
# plt.show()


#Bidirectional Long Short-Term Memory (BiLSTM)
print("\n\n-----------------------BiLSTM---------------------\n\n")
Bi_LSTM_w2v_model = Sequential([
    Embedding(input_dim =embedding_matrix.shape[0], input_length=max_len, output_dim=embedding_matrix.shape[1],weights=[embedding_matrix], trainable=False),
    SpatialDropout1D(0.5),
    Bidirectional(LSTM(25, return_sequences=True)), 
    BatchNormalization(),
    Dropout(0.5),
    GlobalMaxPool1D(),
    Dense(50, activation = 'relu'),
    Dense(numOfClasses, activation = 'sigmoid')
])
Bi_LSTM_w2v_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=[tfa.metrics.F1Score(num_classes=numOfClasses,average='weighted'),'binary_accuracy','accuracy', AUC()])
Bi_LSTM_w2v_model.summary()

if dataset_selection == "4":
    Bi_LSTM_w2v_model_fit = Bi_LSTM_w2v_model.fit(X_tra, y_tra, class_weight=class_weights_dict,batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early])
else:
    Bi_LSTM_w2v_model_fit = Bi_LSTM_w2v_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early])


# # Plot training & validation accuracy values
# plt.plot(Bi_LSTM_w2v_model_fit.history['binary_accuracy'])
# plt.plot(Bi_LSTM_w2v_model_fit.history['val_binary_accuracy'])
# plt.title('Bidirecitonal LSTM-wprd2vec Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(Bi_LSTM_w2v_model_fit.history['loss'])
# plt.plot(Bi_LSTM_w2v_model_fit.history['val_loss'])
# plt.title('Bidirecitonal LSTM-w2v Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Training Loss', 'Validation Loss'], loc='lower right')
# plt.show()

#Gated Recurrent (GRU)
print("\n\n-----------------------GRU---------------------\n\n")
sequence_input = Input(shape=(max_len, ))
GRU_w2v_model = Embedding(input_dim =embedding_matrix.shape[0], input_length=max_len, output_dim=embedding_matrix.shape[1],weights=[embedding_matrix], trainable=False)(sequence_input)
GRU_w2v_model = SpatialDropout1D(0.2)(GRU_w2v_model)
GRU_w2v_model = GRU(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1)(GRU_w2v_model)
GRU_w2v_model = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(GRU_w2v_model)
avg_pool = GlobalAveragePooling1D()(GRU_w2v_model)
max_pool = GlobalMaxPooling1D()(GRU_w2v_model)
GRU_w2v_model = concatenate([avg_pool, max_pool]) 
preds = Dense(numOfClasses, activation="sigmoid")(GRU_w2v_model)
GRU_w2v_model = Model(sequence_input, preds)
GRU_w2v_model.compile(loss='binary_crossentropy',optimizer="adam",metrics=[tfa.metrics.F1Score(num_classes=numOfClasses,average='weighted'),'binary_accuracy','accuracy', AUC() ])
GRU_w2v_model.summary()

if dataset_selection == "4":
    GRU_w2v_model_fit = GRU_w2v_model.fit(X_tra, y_tra, class_weight=class_weights_dict,batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early])
else:
    GRU_w2v_model_fit = GRU_w2v_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early])

# # Plot training & validation accuracy values
# plt.plot(GRU_w2v_model_fit.history['binary_accuracy'])
# plt.plot(GRU_w2v_model_fit.history['val_binary_accuracy'])
# plt.title('Gated Recurrent Unit (GRU) with word2vec Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(GRU_w2v_model_fit.history['loss'])
# plt.plot(GRU_w2v_model_fit.history['val_loss'])
# plt.title('Bidirectional Gated Recurrent Unit (GRU) with word2vec Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Training Loss', 'Validation Loss'], loc='lower right')
# plt.show()

def prediction(text,model_name):
    if model_name=="1":
        model=CNN_w2v_model
    elif model_name=="2":
        model=RNN_w2v_model
    elif model_name=="3":
        model=LSTM_w2v_model
    elif model_name=="4":
        model=Bi_LSTM_w2v_model
    elif model_name=="5":
        model=GRU_w2v_model
    arr=[]
    arr.append(text)
    tokenizer.fit_on_texts(text)
    text = tokenizer.texts_to_sequences(text)
    text = pad_sequences(text, max_len)
    pre=model.predict(text)
    predictions=pre[0]

    class_names=multilabel.classes_
    d={}

    for i in range(0,len(class_names)):
        d[class_names[i]]=predictions[i]
      
    sorted_predictions = dict(sorted(d.items(), key=lambda x: x[1]))

    for key, value in list(sorted_predictions.items())[-6:]:
        print(key, np.format_float_positional(np.float32(value))) 

while(1):
    print("\n------------Testing---------------\n")
    text = input("Enter statement to detect context or enter 'E' to exit:").lower()
    if text == "e":
        break
    else:
        model_name=input("1 to select CNN\n"
                         "2 to select RNN\n"
                         "3 to select LSTM\n"
                         "4 to select BiLSTM\n"
                         "5 to select GRU\n"
                         "Your choice:")
        
        prediction(text,model_name)