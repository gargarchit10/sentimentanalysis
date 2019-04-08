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
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_fscore_support



data = pd.read_csv("test.csv")
text_data = data['Tweet']
text_classes = data['classes']


trainDF = pandas.DataFrame()
trainDF['text'] = text_data
trainDF['label'] = text_classes

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(text_data, text_classes)


# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)


data['dum_val']=1
train_p=data[['tweet_id','classes','dum_val']].pivot_table(index='tweet_id',columns='classes')
train_p=pd.DataFrame(train_p.to_records())
train_p.columns=['tweet_id','negative','neutral','positive']
train_new=pd.merge(data,train_p,on='tweet_id',how='left')
train_new=train_new.drop(['classes','dum_val'],axis=1)
train_2=train_new[["negative","neutral","positive"]].fillna(0)



list_classes = ["negative", "positive", "neutral"]
y = train_2[list_classes].values

ytrain=y[0:3124]
ytest=y[3124:]



embed_size = 300 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use



tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_x.values))
list_tokenized_train = tokenizer.texts_to_sequences(train_x.values)
list_tokenized_test = tokenizer.texts_to_sequences(valid_x.values)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


#print(X_te.shape)



def get_coefs(word,*arr): return word, numpy.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open('data/wiki-news-300d-1M.vec',encoding="utf8"))



word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = numpy.random.normal(0.020940498, 0.6441043, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i-1] = embedding_vector

'''
def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    return metrics.accuracy_score(predictions, ytest), predictions

'''


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
#ylayer=numpy.asarray(ylayer)
x = Dense(3, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


X_t[X_t == lx] = lx-1
X_te[X_te == lx]=lx-1

model.fit(X_t, ytrain , batch_size=32, epochs=2, validation_split=0.1)

'''
inp2 = Input(shape=(maxlen,))
x2 = Embedding(lx, embed_size, weights=[embedding_matrix])(inp2)
x2 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu' , weights=model.layers[2].get_weights() )(x2)
x2 = MaxPooling1D(pool_size=2)(x2)
x2 = Bidirectional(LSTM(300, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, weights=model.layers[4].get_weights()))(x2)
x2 = GlobalMaxPool1D()(x2)
x2 = Dense(300, activation="relu" , weights=model.layers[6].get_weights())(x2)
x2 = Dense(100, activation="relu" , weights=model.layers[7].get_weights())(x2)
model2 = Model(inputs=inp2, outputs=x2)
#model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

activations = model2.predict(X_t)
#model2.fit(X_t, ytrain, batch_size=3124, epochs=1 )

feature=numpy.asarray(activations)


'''

scores = model.evaluate(X_te, ytest, verbose=0)
print("Accuracy on Test Data LSTM: %.2f%%" % (scores[1]*100))


'''
model1 = BinaryRelevance(linear_model.LogisticRegression())
model2 = BinaryRelevance(GaussianNB())
model3=BinaryRelevance(RandomForestClassifier())
model4=BinaryRelevance(AdaBoostClassifier())
model5=BinaryRelevance(DecisionTreeClassifier())
model6=BinaryRelevance(XGBClassifier())
model7=BinaryRelevance(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0))
model8=BinaryRelevance(CatBoostClassifier(iterations=2, learning_rate=1, depth=2, loss_function='MultiClass'))
modelvoting = VotingClassifier(estimators=[('lr', model1), ('gnb', model2), ('rf', model3), ('ab', model4), ('dt', model5), ('xgb', model6), ('gb', model7), ('cb', model8)], voting='hard')
modelvoting.fit(feature,ytrain)
ac=modelvoting.score(X_te,ytest)
print ("Accuracy of Voting Based eEnsemble Model  ", ac)


accuracy ,pred1 = train_model(BinaryRelevance(linear_model.LogisticRegression()), feature, ytrain, X_te)
print ("Accuracy of Logistic Regression ", accuracy)

accuracy,pred2 = train_model(BinaryRelevance(GaussianNB()), feature, ytrain , X_te)
print("Accuray of Naive Bayes ", accuracy)

accuracy,pred3 = train_model(BinaryRelevance(RandomForestClassifier()), feature, ytrain, X_te)
print ("Accuracy of Random Forest", accuracy)

accuracy ,pred4 = train_model(BinaryRelevance(AdaBoostClassifier()), feature, ytrain, X_te)
print ("Accuracy of AdaBoostClassifier ", accuracy)

accuracy,pred5 = train_model(BinaryRelevance(DecisionTreeClassifier()), feature, ytrain, X_te)
print ("Accuracy of Decision Tree ", accuracy)


accuracy,pred6 = train_model(BinaryRelevance(XGBClassifier()), feature, ytrain, X_te)
print ("Accuracy of XGBClassifier ", accuracy)

accuracy,pred7 = train_model(BinaryRelevance(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)), feature, ytrain, X_te)
print ("Accuracy of GradientBoostingClassifier  ", accuracy)

accuracy,pred8 = train_model(BinaryRelevance(CatBoostClassifier(iterations=2, learning_rate=1, depth=2, loss_function='MultiClass')), feature, ytrain, X_te)
print ("Accuracy of CatBoostClassifier  ", accuracy)
'''

y_test = model.predict([X_te], batch_size=1024, verbose=1)

sample_submission = pd.DataFrame()
sample_submission['negative']=1
sample_submission['positive']=1
sample_submission['neutral']=1
sample_submission[list_classes] = y_test
sample_submission.to_csv('C:/Users/DELL/Desktop/BI-LSTM/CNNandBiLSTMSSTWEET.csv', index=False)
#prec = precision_recall_fscore_support(valid_y, pred, average='macro')
#print ("NB, Count Vectors: (Precision , Recall , F_Score ) : ", prec