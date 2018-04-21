import sys
import time
import os
import sklearn
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix 

start_time=time.time()


def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]    
    all_words = []       
    for mail in emails:    
        with open(mail) as m:
            for i,line in enumerate(m):
                if i == 2:  #Body of rumour is only 3rd line of text file
                    words = line.split()
                    all_words += words
    
    dictionary = Counter(all_words)
    
    list_to_remove = list(dictionary.keys())
    for item in range(len(list_to_remove)):
        if list_to_remove[item].isalpha() == False: 
            del dictionary[list_to_remove[item]]
        elif len(list_to_remove[item]) == 1:
            del dictionary[list_to_remove[item]]
    dictionary = dictionary.most_common(3000)
    
    return dictionary


def extract_features(mail_dir): 
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1     
    return features_matrix


# Create a dictionary of words with its frequency

train_dir = 'train-rumours'
dictionary = make_Dictionary(train_dir)

# Prepare feature vectors per training mail and its labels

train_labels = np.zeros(702)
train_labels[351:701] = 1
train_matrix = extract_features(train_dir)

# Training SVM and Naive bayes classifier

model1 = MultinomialNB()
model2 = LinearSVC()
model1.fit(train_matrix,train_labels)
model2.fit(train_matrix,train_labels)

# Test the test rumour for rumours
test_dir = 'test-rumours'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(263)
test_labels[130:263] = 1
result1 = model1.predict(test_matrix)
result2 = model2.predict(test_matrix)



for i in range(len(test_labels)):
    if(result1[i]==0):
        print(i+1," is not a rumour\n")
    else:
        print(i+1," is a rumour\n",)
        
        
tt=time.time()-start_time

print("total time taken to run the progran is",str(tt/60)+'min')

print("accuracy with model 1",sklearn.metrics.accuracy_score(test_labels, result1, normalize=True, sample_weight=None)*100)
print()
print("accuracy with model 2",sklearn.metrics.accuracy_score(test_labels, result2, normalize=True, sample_weight=None)*100)

