import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from datetime import datetime, date
from math import ceil

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


project_folder = '/Users/chaaya/Projects/PredictFutureSales/'

def loadData(project_folder):
    # load training and testing data 
    data_train = pd.read_csv(project_folder+'data/tidy/sales_train_engin.csv')
    data_test = pd.read_csv(project_folder+'data/raw/test.csv')
    return(data_train,data_test)

def decomposeTimeSeries(project_folder,data_train):
    print("Decomposing data! Seasonality, trend and residuals plot can be found in figures folder")
    # this function extract trend, seasonality and residuals from the aggregated data (not differencing between shop_id,item_id etc.)
    data_train_grouped_sales = data_train.groupby(["date_block_num"])["item_cnt_day"].sum()
    # additive criteria
    data_train_grouped_sales_decomposed = sm.tsa.seasonal_decompose(data_train_grouped_sales.values,period=12,model="additive")
    data_train_grouped_sales_decomposed.plot()
    plt.savefig(project_folder+'figures/total_decomposed_additive.pdf')

    # multiplicative criteria
    data_train_grouped_sales_decomposed = sm.tsa.seasonal_decompose(data_train_grouped_sales.values,period=12,model="multiplicative")
    data_train_grouped_sales_decomposed.plot()
    plt.savefig(project_folder+'figures/total_decomposed_multiplicative.pdf')

def testStationarity(data_train):
    # this function test stationarity with Dickey-Fuller test
    data_train_grouped_sales = data_train.groupby(["date_block_num"])["item_cnt_day"].sum()
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(data_train_grouped_sales, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

def SARIMA(project_folder,data_train):
    # this function apply the actual SARIMA model
    data_train_grouped_sales = data_train.groupby(["date_block_num"])["item_cnt_day"].sum()
    def diagnosticPlot(project_folder,data_train_grouped_sales):
        # diagnostic plots to understand the best approach to differentiate data before modelling
        plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
        # Original Series
        fig, axes = plt.subplots(3, 2, sharex=True)
        axes[0, 0].plot(data_train_grouped_sales); axes[0, 0].set_title('Original Series')
        plot_acf(data_train_grouped_sales, ax=axes[0, 1])

        # 1st Differencing
        axes[1, 0].plot(data_train_grouped_sales.diff(periods=12)); axes[1, 0].set_title('1st Order Differencing')
        plot_acf(data_train_grouped_sales.diff().dropna(), ax=axes[1, 1])

        # 2nd Differencing
        axes[2, 0].plot(data_train_grouped_sales.diff(periods=12).diff()); axes[2, 0].set_title('2nd Order Differencing')
        plot_acf(data_train_grouped_sales.diff().diff().dropna(), ax=axes[2, 1])
        plt.savefig(project_folder+'figures/differencing_autocorellation.pdf')

    def model(data_train_grouped_sales):
        grouped_sales_deseason =data_train_grouped_sales.diff(periods=12)
        # 1,1,1 ARIMA Model
        model = smt.SARIMAX(data_train_grouped_sales.values, order=(2,1,1), seasonal_order=(1,0,0,12)).fit()
        print(model.summary())
        return(model)

    def predict(project_folder,model,data_train_grouped_sales):
        sales_results=model.predict(1, len(data_train_grouped_sales)+6, type='levels')
        plt.style.use('seaborn-poster')
        plt.figure()
        plt.plot(list(data_train_grouped_sales), label='Original')
        plt.plot(list(sales_results), ls='--', label='Predicted')
        plt.legend(loc='best')
        plt.title('SARIMA model')
        plt.savefig(project_folder+"figures/SARIMA_prediciton.pdf")
    
    diagnosticPlot(project_folder,data_train_grouped_sales)
    SARIMA_model = model(data_train_grouped_sales)
    predict(project_folder,SARIMA_model,data_train_grouped_sales)

if __name__ == "__main__":
    data_train,data_test = loadData(project_folder)
    decomposeTimeSeries(project_folder,data_train)
    testStationarity(data_train)
    SARIMA(project_folder,data_train)



