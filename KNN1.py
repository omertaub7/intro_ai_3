import matplotlib.pyplot as plt
import pandas as pn
import numpy.matlib as np


def normalize(train: pn.DataFrame, test: pn.DataFrame):
    for (columnName, columnData) in train.iteritems():
        max, min = train[columnName].max(), train[columnName].min()
        train[columnName] = ((train[columnName] - train[columnName].min()) / (train[columnName].max() - train[columnName].min()))
        test[columnName] = ((test[columnName] - min) / (max - min))


def auclidic_distance(x: pn.DataFrame, y: pn.DataFrame) -> float:
    sum = 0
    params = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
              'Age']
    for p in params:
        sum += abs(x[p] - y[p]) ** 2
    return np.sqrt(sum)


# Read Train  and file


train_data_frame = pn.read_csv('train.csv')
test_data_frame = pn.read_csv('test.csv')

normalize(train_data_frame, test_data_frame)


test_result = []
k = 9
for _, test_row in test_data_frame.iterrows():
    dists = []
    for _, train_row in train_data_frame.iterrows():
        dists.append((auclidic_distance(train_row, test_row), train_row['Outcome']))
    dists.sort(key=lambda tup: tup[0])
    knn_9 = dists[0:k]
    sum = 0
    for i in knn_9:
        sum += i[1]
    test_result.append(1 if sum > k / 2 else 0)

# Calculate confustion matrix
conf_mat = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
for (val, real_val) in zip(test_result, test_data_frame['Outcome'].tolist()):
    if val == real_val and val == 1:
        conf_mat['TP'] += 1
    elif val == real_val and val == 0:
        conf_mat['TN'] += 1
    elif val != real_val and val == 0:
        conf_mat['FN'] += 1
    elif val != real_val and val == 1:
        conf_mat['FP'] += 1
    else:
        assert(False)

print(conf_mat)
