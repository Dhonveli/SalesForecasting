import pandas as pd
import numpy as np
from datetime import datetime, date

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

####################################
## Load data, feature engineering ##
####################################

#sales_train = pd.read_csv("/Users/chaaya/Projects/PredictFutureSales/competitive-data-science-predict-future-sales/sales_train.csv")
#test = pd.read_csv("/Users/chaaya/Projects/PredictFutureSales/competitive-data-science-predict-future-sales/test.csv")
#item_categories = pd.read_csv("/Users/chaaya/Projects/PredictFutureSales/competitive-data-science-predict-future-sales/item_categories.csv")
#items = pd.read_csv("/Users/chaaya/Projects/PredictFutureSales/competitive-data-science-predict-future-sales/items.csv")
#shops = pd.read_csv("/Users/chaaya/Projects/PredictFutureSales/competitive-data-science-predict-future-sales/shops.csv")

project_folder_path = '/Users/chaaya/Projects/PredictFutureSales'

def loadData(project_folder_path):
    sales_train = pd.read_csv(project_folder_path+'/data/raw/sales_train.csv')
    test = pd.read_csv(project_folder_path + '/data/raw/test.csv')
    item_categories = pd.read_csv(project_folder_path+'/data/raw/item_categories.csv')
    items = pd.read_csv(project_folder_path+'/data/raw/items.csv')
    shops = pd.read_csv(project_folder_path+'/data/raw/shops.csv')
    # aggregate all data
    sales_train_item = sales_train.merge(items,how="left",on="item_id")
    sales_train_item_categ = sales_train_item.merge(item_categories,how="left",on="item_category_id")
    sales_train_item_categ_shop = sales_train_item_categ.merge(shops,how="left",on="shop_id")
    return(sales_train_item_categ_shop)

def featureEngineering(data):
    data['month'] = data.date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y').strftime('%m'))
    data['year'] = data.date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y').strftime('%Y'))
    data['day'] = data.date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y').strftime('%d'))

    data['city'] = data['shop_name'].str.split(' ').map(lambda x: x[0])
    data['city_code'] = LabelEncoder().fit_transform(data['city'])
    data['type'] = data['item_category_name'].str.split('-').map(lambda x: x[0].strip())
    data['type_code'] = LabelEncoder().fit_transform(data['type'])
    data['subtype'] = data['item_category_name'].str.split('-').map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
    data['subtype_code'] = LabelEncoder().fit_transform(data['subtype'])
    return(data)

def saveFile(data,project_folder_path):
    data.to_csv(project_folder_path+'/data/tidy/sales_train_engin.csv')


def processRawData(project_folder_path):
    data = loadData(project_folder_path)
    data_engin = featureEngineering(data)
    saveFile(data_engin,project_folder_path)

processRawData(project_folder_path)

