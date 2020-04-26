import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from dateutil.easter import easter
from fbprophet import Prophet


# NOT TESTED

project_folder = '/Users/chaaya/Projects/PredictFutureSales/'

def loadData(project_folder):
    data_train = pd.read_csv(project_folder+'data/tidy/sales_train_engin.csv')
    data_test = pd.read_csv(project_folder+'data/raw/test.csv')
    return(data_train,data_test)

def Prophet(project_folder,data_train):
    # Apply Facebook Prophet
    data_train_grouped_sales = data_train.groupby(["date_block_num"])["item_cnt_day"].sum()
    data_train_grouped_sales.columns=['ds','y']
    def model(data_train_grouped_sales):
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True) 
        model.fit(data_train_grouped_sales)
        return(model)
    def predict(project_folder,model,data_train_grouped_sales):
        future = model.make_future_dataframe(periods = 6, freq = 'MS')  
        sales_results = model.predict(future)
        sales_results[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        # plotting
        plt.style.use('seaborn-poster')
        plt.figure()
        plt.plot(list(data_train_grouped_sales), label='Original')
        plt.plot(sales_results['yhat'], ls='--', label="Predicted")
        plt.legend(loc='best')
        plt.title('FB Prophet model')
        plt.show()


if __name__ == "__main__":
    data_train, data_test = loadData(project_folder)
    Prophet(project_folder,data_train)
