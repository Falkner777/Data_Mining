
import numpy as np
import pandas as pd

import string 
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec,utils
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv("amazon.csv")



def wordCleaning(df):
    """
    1. We create a new dataframe with the column "text"
    2. We remove stopwords from the text
    3. We remove punctuation from the text
    4. We remove numbers from the text
    5. We remove extra spaces from the text
    6. We apply simple preprocessing to the text
    
    :param df: The dataframe that contains the text column
    :return cleanData: the preprocessed dataset
    """
    cleanData = pd.DataFrame({"text":df["Text"]})
    stop_words = set(stopwords.words('english'))
    punct = re.compile('[%s]' % re.escape(string.punctuation))
  
    cleanData["text"] = cleanData["text"].apply(lambda x : x.lower())
    cleanData["text"] = cleanData["text"].apply(lambda x : " ".join([w for w in x.split() if not w in stop_words]))
    cleanData["text"] = cleanData["text"].apply(lambda x : re.sub(r'\'[\w]+[^\S]'," ",x))
    cleanData["text"] = cleanData["text"].apply(lambda x : re.sub(r'[0-9]',"",x))
    cleanData["text"] = cleanData["text"].apply(lambda x : punct.sub(' ', x))
    cleanData["text"] = cleanData["text"].apply(lambda x : re.sub(r'\s'," ",x))
    cleanData["text"] = cleanData["text"].apply(lambda x : " ".join(x.split()))
    cleanData["text"] = cleanData["text"].apply(lambda x: utils.simple_preprocess(x))
    
    return cleanData

def make_feature_vec(words, model, num_features):
    """
    For each word in the review, if the word is in the model's vocabulary, add its feature vector to the
    total. Then, divide the result by the number of words to get the average
    
    :param words: a list of words
    :param model: the Word2Vec model we're using
    :param num_features: The number of features to be used in the model
    :return: The average of the word vectors for each word in the review.
    """
  
    feature_vec = np.zeros((num_features,),dtype="float32")  # pre-initialize (for speed)
    nwords = 0.
    index2word_set = model.wv.key_to_index.keys()  # words known to the model
    
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            feature_vec = np.add(feature_vec,model.wv[word])

    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec

def avg_reviews(df, model, num_features):
    """
    It takes a dataframe, a word2vec model, and a number of features, and returns a dataframe with a new
    column called 'vectorised' which contains the word2vec representation of each review.
    
    :param df: the dataframe containing the text
    :param model: the word2vec model
    :param num_features: The number of features to be used in the model
    :return: A dataframe with the vectorised text
    """
    new = pd.DataFrame({"vectorised":df['text']})
    
    new['vectorised'] = new['vectorised'].apply(lambda x: make_feature_vec(x, model, num_features))

    return new

# read the dataset and apply cleaning
df = pd.read_csv("amazon.csv")
cleanDF = wordCleaning(df)

# create a word embeddings model that will vectorize each word into float numbers
vec_size = 100
model = Word2Vec(vector_size=vec_size,window= 15, min_count= 2, workers=6)
model.build_vocab(cleanDF["text"], progress_per=1000)
model.train(cleanDF["text"], total_examples=model.corpus_count, epochs=model.epochs)

# for every word in a sentence get the average and put them in a dataframe
vectored = avg_reviews(cleanDF,model, vec_size)
vectored['vecorised'] = vectored['vectorised'].apply(lambda x: x.astype(np.float128))

train_Frame = pd.DataFrame(vectored['vectorised'].tolist())
# remove all the nan indexes
nanIndexes = train_Frame[train_Frame.isna().any(axis=1)].index
train_Frame.drop(index=nanIndexes,inplace=True)
df.drop(index=nanIndexes, inplace=True)

#split the data into train test set
X_train, X_test, y_train, y_test = train_test_split(train_Frame, df["Score"], test_size=0.2, random_state=1)

#create a random forest algorithm and run the model
randomForest = RandomForestClassifier(n_estimators=200, criterion='gini',n_jobs=4)
randomForest.fit(X_train, y_train)

y_pred = randomForest.predict(X_test)
print("--------Results Using The Initial Classes--------")
print('Accuracy: %.3f' % precision_score(y_test, y_pred,average='micro'))
print('Recall %.3f' % recall_score(y_test,y_pred, average='micro'))
print('F1_Score %.3f' % f1_score(y_test,y_pred, average='micro'))


df.loc[df["Score"] == 1,"Score"] = 0
df.loc[df["Score"] == 2,"Score"] = 0
df.loc[df["Score"] == 4,"Score"] = 2
df.loc[df["Score"] == 5,"Score"] = 2
df.loc[df["Score"] == 3,"Score"] = 1

X_train, X_test, y_train, y_test = train_test_split(train_Frame, df["Score"], test_size=0.2, random_state=1)

randomForest = RandomForestClassifier(n_estimators=200, criterion='gini',n_jobs=4)
randomForest.fit(X_train, y_train)

y_pred = randomForest.predict(X_test)
print("--------Results Using Three Classes--------")
print('Accuracy: %.3f' % precision_score(y_test, y_pred,average='micro'))
print('Recall %.3f' % recall_score(y_test,y_pred, average='micro'))
print('F1_Score %.3f' % f1_score(y_test,y_pred, average='micro'))


