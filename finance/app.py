import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import datetime as dt

@st.cache
def load_data():
    components = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return components.set_index('Symbol')

def load_quotes(asset):
    return yf.download(asset)

def create_model(data):
    #prepare the data
    data['Date'] = data.index
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date']=data['Date'].map(dt.datetime.toordinal)
    X = data['Date'].values.reshape(-1,1)
    y = data['Close'].values.reshape(-1,1)
    
    #split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #train the algorithm
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)

    return regressor, X_test

def main():
    components = load_data()
    title = st.empty()

    def label(symbol):
        return symbol

    asset = st.sidebar.selectbox('Select asset', components.index.sort_values(), index=3, format_func=label)
    title.title(asset)

    data0 = load_quotes(asset)
    if data0 is not None:
        data = data0.copy().dropna()
        data.index.name = None

        # Adding Linear Regression prediction
        if st.sidebar.checkbox('Predict closing price with Linear Regression'):
            model, X_test = create_model(data)
            y_pred = model.predict(X_test)
            
            #visualize the results
            plt.figure(figsize=(10,5))
            plt.scatter(X_test, y_test,  color='gray')
            plt.plot(X_test, y_pred, color='red', linewidth=2)
            plt.show()

if __name__ == '__main__':
    main()
