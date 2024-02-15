import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from collections import Counter

X_train = []
Y_train = []

min_token_len = 4
max_token_len = 6

classifier1 = Pipeline([
    ('vectorizer', CountVectorizer(min_df=1, max_df=8)),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))
])

classifier2 = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB()),
])

def cleanup_string(input_string):
    cleand = re.sub(r'[ ;|,-]', '', input_string)
    return cleand

def split_string_with_regex(cleand_string, len):
    return re.findall('.{1,'+str(len)+'}', cleand_string)


def add_to_learning_set(tokens, name):
    if (len(tokens)>0 and (len(tokens[-1]) < min_token_len)):
        tokens.pop()
    seq = ' '.join(tokens).upper()
    print(seq)
    X_train.append(seq)
    Y_train.append(name)

def process_file(name):
    fname = ".\\data\\"+name+".txt"
    print('Processing:'+fname)
    f = open(fname, "r")
    for x in f:
        if (x[0]=='#'):
            break
        cx = cleanup_string(x)
        if (len(cx)>min_token_len):
            print(f'cleand: {cx}')
            for i in range(min_token_len, max_token_len+1):
                print(f'Size:{i}')
                tokens = split_string_with_regex(cx,i)
                add_to_learning_set(tokens, name)
                tokens = split_string_with_regex(cx[1:len(cx)-1],i)
                add_to_learning_set(tokens, name)


process_file("abogi")
#process_file("mohanam")
#process_file("khamas")

classifier1.fit(X_train, Y_train)
classifier2.fit(X_train, Y_train)

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

def predict(test_data):
    X_test = []
    for x in test_data:
        for i in range(min_token_len, max_token_len+1):
            x_tokes = split_string_with_regex(x,i)
            if len(x_tokes[-1]) < 4:
                x_tokes.pop()
            seq = ' '.join(x_tokes).upper()
            X_test.append(seq)

    predicted1 = classifier1.predict(X_test)
    predicted2 = classifier2.predict(X_test)
    
    raga = []
    for item, raga1, raga2 in zip(X_test, predicted1, predicted2):
        raga.append(raga1)
        raga.append(raga2)
        print(f"{item} => {raga1} or {raga2}")

    return [most_frequent(raga)]

if __name__ == '__main__':
    test_data = np.array([
        "srgpgpds",
        "MPDPPMGM",
        "mndnsmgm"
    ])
    print(predict(test_data))