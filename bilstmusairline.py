from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.ensemble import RandomForestClassifier
import pandas, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.ensemble import RandomForestClassifier
import pandas, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.metrics import precision_recall_fscore_support
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing import sequence
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import precision_recall_fscore_support
import re
from sklearn.utils import shuffle

data= pd.read_csv("usairline.csv" , encoding='utf-8')
data=data[['_unit_id','text','airline_sentiment']]
data = data.dropna()

text_data=data['text']
text_classes=data['airline_sentiment']

# Removing Special Characters
text_data = text_data.str.replace('\W ', '')
text_classes = text_classes.str.replace('\W ', '')

# Removing Numbers
text_data = text_data.str.replace('\d+ ', '')
text_classes = text_classes.str.replace('\d+ ', '')

data['text'] = text_data
data['airline_sentiment'] = text_classes



train_x, valid_x, train_y, valid_y = model_selection.train_test_split(text_data, text_classes)


# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)


data['dum_val']=1
data['tweet_id'] = data.index
data.sort_index(inplace=True)

train_p=data[['tweet_id','airline_sentiment','dum_val',]].pivot_table(index='tweet_id',columns='airline_sentiment')
train_p=pd.DataFrame(train_p.to_records())
train_p.columns=['tweet_id','negative','neutral','positive']
train_new=pd.merge(data,train_p,on='tweet_id',how='left')
train_new=train_new.drop(['airline_sentiment','dum_val'],axis=1)
train_2=train_new[["negative","neutral","positive"]].fillna(0)
list_classes = ["negative", "positive", "neutral"]

y = train_2[list_classes].values

ytrain=y[0:3030]
ytest=y[3030:]

embed_size = 300 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use





tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_x.values))
list_tokenized_train = tokenizer.texts_to_sequences(train_x.values)
list_tokenized_test = tokenizer.texts_to_sequences(valid_x.values)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


def get_coefs(word,*arr): return word, numpy.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open('data/wiki-news-300d-1M.vec',encoding="utf8"))




word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = numpy.random.normal(0.020940498, 0.6441043, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i-1] = embedding_vector


lx=len(embedding_matrix)
inp = Input(shape=(maxlen,))
x = Embedding(lx, embed_size, weights=[embedding_matrix])(inp)
x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Bidirectional(LSTM(300, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(300, activation="relu")(x)
x = Dense(100, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(3, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_t[X_t == lx] = lx-1
X_te[X_te == lx]=lx-1


model.fit(X_t, ytrain , batch_size=32, epochs=2, validation_split=0.1)

scores = model.evaluate(X_te, ytest, verbose=0)
print("Accuracy on Test Data : %.2f%%" % (scores[1]*100))

'''

y_test = model.predict([X_te], batch_size=1024, verbose=1)

sample_submission = pd.DataFrame()
sample_submission['negative']=1
sample_submission['positive']=1
sample_submission['neutral']=1
sample_submission[list_classes] = y_test
sample_submission.to_csv('C:/Users/DELL/Desktop/BI-LSTM/CNNandBiLSTMusairline.csv', index=False)
'''

