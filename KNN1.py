import matplotlib.pyplot as plt
import pandas as pn




def normalize(df : pn.DataFrame):


    for (columnName, columnData) in df.iteritems():
        max_val=max(columnData)
        min_val=min(columnData)
        for item in df[columnName]:
            df.at[columnName,item]  =float((item-min_val)/(max_val-min_val))

#Read Train  and file


train_data_frame =pn.read_csv('train.csv')
test_data_frame= pn.read_csv('test.csv')
normalize(train_data_frame)
normalize(test_data_frame)

