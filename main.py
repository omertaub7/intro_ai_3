import matplotlib.pyplot as plt
from sklearn import tree, metrics

import csv

#Read Train file
clf = tree.DecisionTreeClassifier(criterion="entropy")
train_file = open('train.csv')
data = csv.reader(train_file, delimiter=',')
train_data, train_results = [], []
for row in data:
    train_data.append(row[0:8])
    train_results.append(row[8])
train_data, train_results=train_data[1:len(train_data)],train_results[1:len(train_results)]
#Train the tree
clf.fit(train_data,train_results)

sum=0
all=len(train_results)
for r in train_results:
   sum+=int(r)

print(all,sum)
#Load test file
test_file = open('test.csv')
data = csv.reader(test_file, delimiter=',')
test_data,test_results = [], []
for row in data:
    test_data.append(row[0:8])
    test_results.append(row[8])
test_data, test_results=test_data[1:len(test_data)], test_results[1:len(test_results)]
#Test the tree
clf_results = clf.predict(test_data)
#Generate prediction tree
conf_mat1=metrics.confusion_matrix(test_results,clf_results)
conf_mat1[0][0],conf_mat1[1][1]=conf_mat1[1][1],conf_mat1[0][0]
print(conf_mat1)



#Now we are going to prune the tree
for val in [3, 9, 27]:
    clf = tree.DecisionTreeClassifier(criterion="entropy",min_samples_split=val)
    clf.fit(train_data,train_results)
    print("~~~~~~~~~ X =", val, "~~~~~~~~~")
    clf_results = clf.predict(test_data)
    conf_mat2 = metrics.confusion_matrix(test_results, clf_results)
    conf_mat2[0][0], conf_mat2[1][1] = conf_mat2[1][1], conf_mat2[0][0]
    print(conf_mat2)
    #plt.figure(figsize=(40, 20))
    #_ = tree.plot_tree(clf, filled=True,
    #                   feature_names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
    #                                  'DiabetesPedigreeFunction', 'Age', 'Outcome'])
    #plt.show()

print(conf_mat1[0][1]+4*conf_mat1[1][0] ,"    (without weight: )",  conf_mat1[0][1]+conf_mat1[1][0] )
print(conf_mat2[0][1]+4*conf_mat2[1][0], "    (without weight: )", conf_mat2[0][1]+conf_mat2[1][0])


'''
    dot_data = StringIO()
    #export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True)
    #graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    #Image(graph.create_png())


plt.figure(figsize=(40,20))
_=tree.plot_tree(clf,filled=True,feature_names=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'])
plt.show()
'''
