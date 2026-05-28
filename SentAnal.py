### WORK IN PROGRESS

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from spellchecker import SpellChecker
spell = SpellChecker(language='en') # Vous pouvez changer la langue

nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 


df = pd.read_csv('training_1600000.csv', encoding='latin-1')

print(df.keys())

df['polarity of tweet'].unique()

df[df.keys()[0]] = df[df.keys()[0]].astype("string")


df.loc[df[df.keys()[0]] == '4', df.keys()[0]] = '1' #'Positive'
df.loc[df[df.keys()[0]] == '0', df.keys()[0]] = '0' #'Negative'

df['polarity of tweet'] = pd.to_numeric(df['polarity of tweet'])

print(df['polarity of tweet'].value_counts())
def clean_text(text, stop_words, spell):
    text = re.sub(r'@\w+', ' Nom_uti ', text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www.\S+', ' ', text)
    
    text = text.replace(":@", " confused ")
    text = text.replace(";)", " winking ")
    text = text.replace(";p", " winking ")
    text = text.replace(":)", " happy ")
    text = text.replace("-_-", " annoyance ")
    text = text.replace(":|", " resignation ")
    text = text.replace(":/", " resignation ")
    text = text.replace(":S", " sceptic ")
    text = text.replace(":O", " suprised ")
    text = text.replace(":-D", " happy ")
    text = text.replace("=D", " happy ")
    text = text.replace(":-P", " happy ")
    text = text.replace("=(", " sad ")
    text = text.replace(":(", " happy ")
    text = text.replace("xD", " excited ")
    text = text.replace("XD", " excited ")
    text = text.replace("O.o", " shocked ")
    text = text.replace("2nite"," tonight ")
    text = text.replace("â™¥"," heart ")
    text = text.replace("&gt;"," awkward ")
    text = text.replace("&lt;3"," heart ")
    text = text.replace("&amp;"," and ")
    text = text.replace("plz"," please ")
    

    text = re.sub(r'(?<=[\sa-zA-Z])2(?=[\sa-zA-Z])', r' to ', text)
    text = text.replace("hr", " hours ")
    text = text.replace("hrs", " hours ")
    text = text.replace("...", " ELLIPSIS ")
    text = text.replace("?"," Intero ")
    text = text.replace("'"," ' ")
    text = text.replace("!"," Excla ")
    text = text.replace(".", " ")
    text = text.replace("@ ", " at ")
    text = text.replace(",", " ")
    text = text.replace("-", " ")
    text = text.replace("/", " ")
    text = text.replace(":", " ")
    text = text.replace(";", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("[", " ")
    text = text.replace("]", " ")
    text = text.lower()
    #text = ' '.join(word for word in text.split() ) # if word not in stop_words)
    text = ' '.join(correction(word, spell) for word in text.split() ) # if word not in stop_words)
    return text

def correction(word, spell):
    
    if spell.known([word]):
        #print("test")
        return(word)
    corrected = re.sub(r'([a-zA-Z])\1{3,}', r'\1\1', word)
    if spell.known([corrected]):
        return(corrected)
    corrected = re.sub(r'([a-zA-Z])\1{1,}', r'\1', corrected)
    
    # word_correction = spell.correction(word)
    
    # if word_correction :
    #     return word_correction
    
    
    return corrected
        
        
        
## Modele
from sklearn.model_selection import train_test_split

X = df[df.keys()[5]]
y = df[df.keys()[0]]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3333)

train_tweets_cleaned = [clean_text(tweet, stop_words, spell) for tweet in X_train]
test_tweets_cleaned = [clean_text(tweet, stop_words, spell) for tweet in X_test]
        
## Tokenization
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(train_tweets_cleaned)

train_sequences = tokenizer.texts_to_sequences(train_tweets_cleaned)
test_sequences = tokenizer.texts_to_sequences(test_tweets_cleaned)

## Padding
from tensorflow.keras.preprocessing.sequence import pad_sequences

maxlen  = 0
for i in train_sequences:
    if len(i)>maxlen :
        maxlen  = len(i)

train_padded = pad_sequences(train_sequences, padding='post', maxlen=maxlen, truncating='post')
test_padded = pad_sequences(test_sequences, padding='post', maxlen=maxlen, truncating='post')


## Construction du modele
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam

vocab_size = len(tokenizer.word_index) + 1 

embedding_dim = 100
lstm_units = 128

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=maxlen),
    Bidirectional(LSTM(lstm_units, return_sequences=True)),
    Dropout(0.33),
    LSTM(int(lstm_units/2)),
    Dense(1, activation='sigmoid') # Couche de sortie avec une seule unité et activation sigmoid pour classification binaire (0 ou 1)
])

custom_adam = Adam(learning_rate=0.001)
model.compile(optimizer=custom_adam,
              loss='binary_crossentropy', # Fonction de perte pour classification binaire
              metrics=['accuracy'])

model.summary()

epochs = 10 # Nombre d'époques d'entraînement
batch_size = 256 # Taille du lot pour l'entraînement

load=0
new=0
weighted=0
GloVer=0
if load : 
    from keras.models import load_model
    model = load_model("MyModel.h5")
elif weighted:
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train)
    class_weights_dict = {i : class_weights[i] for i in range(len(class_weights))}

    history = model.fit(
        train_padded,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(test_padded, y_test),
        class_weight=class_weights_dict)
elif GloVer:
    import gensim.downloader as api
    glove_embeddings = api.load("glove-wiki-gigaword-100")
    embedding_dim = glove_embeddings.vector_size
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    word_index = tokenizer.word_index
    num=0
    for word, i in word_index.items():
        if i < vocab_size:
            try:
                embedding_matrix[1] = glove_embeddings[word]
            except :
                num=num+1
                print(word, 'pass')
                pass
            
elif new :
    history = model.fit(
        train_padded,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(test_padded, y_test) # Utiliser les données de test comme ensemble de validation
        )

    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()

    plt.show()


    model.save("MyModel.h5")
