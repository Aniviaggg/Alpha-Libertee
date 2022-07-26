from collections import defaultdict
from encodings import utf_8
import os
import random
import string

import cleantext
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from numpy import shape
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pickle
import nltk


BASE_DIR = '' #TO be filled out later
LABELS = ['Beginner', 'Intermediary', 'Experienced']
stop_words = set(stopwords.words('english'))

def create_data_set():
    with open('data.txt','w', encoding = "utf_8") as outerfile:
        for label in LABELS:
            dir = os.path.join(BASE_DIR, label)
            for filename in os.listdir(dir):
                fullfilename = str(os.path.join(dir, filename))
                with open(fullfilename, 'rb') as file:
                    text = file.read().decode(errors='replace').replace('\n', '')
                    outerfile.write('%s\t%s\t%s\n' % (label, filename, text))


def setup_docs():
    docs = []
    with open('data.txt', 'r', encoding = 'utf8') as datafile:
        for row in datafile:
            parts = row.split('\t')
            doc = (parts[0], parts[-1].strip())
            docs.append(doc)
        return docs


def get_splits(docs):
    random.shuffle(docs)

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    pivot = int(.80 * len(docs))

    for i in range(0, pivot):
        X_train.append(docs[i][1])
        y_train.append(docs[i][0])
    
    for j in range(pivot, len(docs)):
        X_train.append(docs[j][1])
        y_train.append(docs[j][0])
    # note that we currently do not have any test data, so the current returned x and y test are empty lists.
    return X_train, X_test, y_train, y_test


def get_tokens(text):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if not t in stop_words]
    return tokens


def print_frequency_dist(docs):
    tokens = defaultdict(list)

    for doc in docs[:-1]:
        doc_label = doc[0]
        doc_text = cleantext.clean(doc[1])
        doc_tokens = get_tokens(doc_text)
        tokens[doc_label].extend(doc_tokens)

    for category_label, category_tokens in tokens.items():
        print(category_label)
        fd = FreqDist(category_tokens)
        print(fd.most_common(20))


def evaluate_classifier(title, classifier, vectorizer, X_test, y_test):
    # currently the x test and y test are empty lists so there will be error messages
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_tfidf)

    precision = metrics.precision_score(y_test, y_pred, average='macro')
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    f1 = metrics.f1_score(y_test, y_pred, average='macro')

    print("%s\t%f\t%f\t%f\n" % (title, precision, recall, f1))


def train_classifier(docs):
    # currently the x test and y test are empty lists so there will be error messages
    X_train, X_test, y_train, y_test = get_splits(docs)

    vectorizer = CountVectorizer(stop_words='english',ngram_range=(1,3),min_df=3, analyzer='word')

    dtm = vectorizer.fit_transform(X_train)

    naive_bayes_classifier = MultinomialNB().fit(dtm, y_train)

    evaluate_classifier("Naive Bayes\tTRAIN\t", naive_bayes_classifier, vectorizer, X_train, y_train)
    # currently the x test and y test are empty lists so there will be error messages
    evaluate_classifier("Naive Bayes\tTEST\t", naive_bayes_classifier, vectorizer, X_test, y_test)

    #store classifier
    clf_filename = 'naive_bayes_classifier.pkl'
    pickle.dump(naive_bayes_classifier, open(clf_filename, 'wb'))

def classify(text):
    clf_filename = 'naive_bayes_classifier.pkl'
    nb_clf = pickle.load(open(clf_filename, 'rb'))

    # currently do not have this file yet since we do not have test data
    '''
    vec_filename = 'count_vectorizer.pkl'
    vectorizer = pickle.load(open(vec_filename,'rb'))

    pred = nb_clf.predict(vectorizer.transform(text))

    print(pred[0])'''


if __name__ == '__main__':
    #Create dataset
    create_data_set()
    print('Done')
    docs = setup_docs()
    print_frequency_dist(docs)

    train_classifier(docs)

    # we comment out the classify part because currently the x test and y test are empty lists so there will be error messages
    '''new_doc = ''

    classify(new_doc)'''