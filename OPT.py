import csv
import itertools
import pandas as pn

train_file = open('train.csv')
data = csv.reader(train_file, delimiter=',')
train_data, train_results = [], []
for row in data:
    train_data.append(row[0:8])
    train_results.append(row[8])
train_data, train_results=train_data[1:len(train_data)],train_results[1:len(train_results)]
#Now, go on all subsets T<=A to choose the best subset of attributes to choose the optimal
test_file = open('test.csv')
data = csv.reader(test_file, delimiter=',')
test_data,test_results = [], []
for row in data:
    test_data.append(row[0:8])
    test_results.append(row[8])
test_data, test_results=test_data[1:len(test_data)], test_results[1:len(test_results)]
#Get all subsets
subsets = []
for i in range(8):
    combi = (set(itertools.combinations({0,1,2,3,4,5,6,7}, i)))
    for x in combi:
        subsets.append(x)

train_data_frame = pn.read_csv('train.csv')
test_data_frame = pn.read_csv('test.csv')
best_subset, best_val = (), 0
from sklearn import neighbors as nb
from sklearn import metrics


def normalize(train: pn.DataFrame, test: pn.DataFrame):
    for (columnName, columnData) in train.iteritems():
        max, min = train[columnName].max(), train[columnName].min()
        train[columnName] = ((train[columnName] - train[columnName].min()) / (
                train[columnName].max() - train[columnName].min()))
        test[columnName] = ((test[columnName] - min) / (max - min))


for s in subsets:
    if s == ():
        continue
    traits_to_remove = {0,1,2,3,4,5,6,7} - set(s)
    check_train = train_data_frame.drop(train_data_frame.columns[list(traits_to_remove)], axis=1)
    check_train = check_train.drop(labels='Outcome', axis=1)
    check_test =  test_data_frame.drop(test_data_frame.columns[list(traits_to_remove)], axis=1)
    check_test = check_test.drop(labels='Outcome', axis=1)
    clf = nb.KNeighborsClassifier(n_neighbors=9)
    normalize(check_train, check_test)
    clf.fit(check_train.values, train_data_frame['Outcome'])
    res = clf.predict(check_test[train_data_frame.columns[list(s)]]).tolist()
    conf_mat = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for (val, real_val) in zip(res, test_data_frame['Outcome'].tolist()):
        if val == real_val and val == 1:
            conf_mat['TP'] += 1
        elif val == real_val and val == 0:
            conf_mat['TN'] += 1
        elif val != real_val and val == 0:
            conf_mat['FN'] += 1
        elif val != real_val and val == 1:
            conf_mat['FP'] += 1

    correct = conf_mat['TP'] + conf_mat['TN']
    if correct > best_val:
        best_subset, best_val = s, correct

print(list(map(lambda x: "ind"+str(x), best_subset)))
