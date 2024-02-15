import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier

X_train = np.array([
'GGRRSSRR GGRRSSRR', "SRSDSRGP SRGPSRGR",
"GPGGRSRG RRSDSRGR",  "GPGPDPDS DPGDPGR",
"SRGRGRDS RSRSPDSD",
"SRGRSNSR SNDNSNDP", "GMPMNDNPD SRSN",
"gmpdn mpdns", "smgmp gmpdn",
"dpdns smgm", "sgrmpnps"
])

y_train = [  
    "Mohanam","Mohanam",
    "Mohanam","Mohanam","Mohanam", 
    "Khamas", "Khamas", "Khamas",
    "Khamas", "Khamas", "Khamas"
]

classifier = Pipeline([
    ('vectorizer', CountVectorizer(min_df=1, max_df=2)),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))
])

classifier.fit(X_train, y_train)

X_test = np.array([
    "srgp dsrg",
    "smgm pdns"
])

def predict(X_test):
    predicted = classifier.predict(X_test)
    for item, labels in zip(X_test, predicted):
        print(f"{item} => {labels}")
        #return predicted

predict(X_test)