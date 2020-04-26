import pandas as pd
import numpy as np
import seaborn as sns
import NestedLSTM
import matplotlib.pyplot as plt

from datetime import datetime, date



project_folder = '/Users/chaaya/Projects/PredictFutureSales/'

def loadData(project_folder):
    # load training and testing data 
    data_train = pd.read_csv(project_folder+'data/tidy/sales_train_engin.csv')
    data_test = pd.read_csv(project_folder+'data/raw/test.csv')
    return(data_train,data_test)

def formatData(data_train,data_test):
    # this function format training and testing data to feed the NN
    # format training
    data_train_grouped_sales = data_train.groupby([data_train.date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y').strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()
    data_train_grouped_sales = data_train_grouped_sales[["date","item_id","shop_id","item_cnt_day"]]
    data_train_grouped_sales = data_train_grouped_sales.pivot_table(index=['item_id','shop_id'], columns='date',values='item_cnt_day',fill_value=0).reset_index()
    # format testing
    data_train_test_merged = pd.merge(data_test, data_train_grouped_sales, on=['item_id','shop_id'], how='left')
    data_train_test_merged = data_train_test_merged.fillna(0)
    data_train_test_merged = data_train_test_merged.drop(labels=['ID', 'shop_id', 'item_id'], axis=1)
    return(data_train_test_merged)

def NestedLSTM(data_train):
    TARGET = '2015-10'
    y_train = data_train[TARGET].as_matrix().reshape(214200, 1)
    X_train = data_train.drop(labels=[TARGET], axis=1).as_matrix().reshape((214200, 33, 1))
    X_train = X_train.as_matrix().reshape((214200, 33, 1))
    



