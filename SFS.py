import csv
import pandas as pn

train_data_frame = pn.read_csv('train.csv')
test_data_frame = pn.read_csv('test.csv')
from sklearn import neighbors as nb

def normalize(train: pn.DataFrame, test: pn.DataFrame):
    for (columnName, columnData) in train.iteritems():
        max, min = train[columnName].max(), train[columnName].min()
        train[columnName] = ((train[columnName] - train[columnName].min()) / (
                train[columnName].max() - train[columnName].min()))
        test[columnName] = ((test[columnName] - min) / (max - min))

best_subset, best_val = [], 0
while True:
    changed = False
    init_subset = best_subset.copy()
    for i in range(8):
        if i in best_subset:
            continue
        subset = init_subset.copy() # make sure for deep copy
        subset.append(i)
        s = tuple(sorted(subset))
        #from now and on, same logic as OPT
        traits_to_remove = {0, 1, 2, 3, 4, 5, 6, 7} - set(s)
        check_train = train_data_frame.drop(train_data_frame.columns[list(traits_to_remove)], axis=1)
        check_train = check_train.drop(labels='Outcome', axis=1)
        check_test = test_data_frame.drop(test_data_frame.columns[list(traits_to_remove)], axis=1)
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

        correct = (conf_mat['TP'] + conf_mat['TN']) / sum(conf_mat.values())
        if correct >= best_val:
            best_subset, best_val = subset.copy(), correct
            changed = True
    if changed == False:
        break

print(sorted(list(map(lambda x: "ind"+str(x), best_subset))))

