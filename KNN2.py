import matplotlib.pyplot as plt
import pandas as pn
import numpy.matlib as np


def knn_2(k : int):
    def normalize(train: pn.DataFrame, test: pn.DataFrame):
        for (columnName, columnData) in train.iteritems():
            max, min = train[columnName].max(), train[columnName].min()
            train[columnName] = ((train[columnName] - train[columnName].min()) / (
                        train[columnName].max() - train[columnName].min()))
            test[columnName] = ((test[columnName] - min) / (max - min))

    def auclidic_distance(x: pn.DataFrame, y: pn.DataFrame) -> float:
        sum = 0
        params = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                  'DiabetesPedigreeFunction',
                  'Age']
        for p in params:
            sum += abs(x[p] - y[p]) ** 2
        return np.sqrt(sum)

    # Read Train  and file

    train_data_frame = pn.read_csv('train.csv')
    test_data_frame = pn.read_csv('test.csv')

    normalize(train_data_frame, test_data_frame)

    test_result = []

    for _, test_row in test_data_frame.iterrows():
        dists = []
        for _, train_row in train_data_frame.iterrows():
            dists.append((auclidic_distance(train_row, test_row), train_row['Outcome']))
        dists.sort(key=lambda tup: tup[0])
        knn_k = dists[0:k]
        pos_sum = 0
        neg_sum=0
        for i in knn_k:
            if i[1]==1:
                pos_sum+=4  # new classifying rule gives 4 times weight for positive examples
            if i[1]==0:
                neg_sum+=1

        test_result.append(1 if pos_sum > neg_sum else 0)

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

    return [[conf_mat['TP'], conf_mat['FP']] ,[conf_mat['FN'], conf_mat['TN']]]


for k in [1,3,9,27]:
    print("--k : " , k)
    conf_matrix = knn_2(k)
    print("[",conf_matrix[0],'\n', conf_matrix[1], "]", sep='')

