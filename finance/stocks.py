#Import libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import seaborn as sns
import datetime as dt

@st.cache
def load_data():
    components = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return components.set_index('Symbol')

def load_quotes(asset):
    return yf.download(asset)

def create_model(data):
    data['Date'] = data.index
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date']=data['Date'].map(dt.datetime.toordinal)
    X = data['Date'].values.reshape(-1,1)
    y = data['Close'].values.reshape(-1,1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)

    return regressor, X_test, y_test

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

        section = st.sidebar.slider('Number of quotes', min_value=30,
                                    max_value=min([2000, data.shape[0]]),
                                    value=500, step=10)

        data2 = data[-section:]['Adj Close'].to_frame('Adj Close')

        sma = st.sidebar.checkbox('SMA')
        if sma:
            period = st.sidebar.slider('SMA period', min_value=5, max_value=500,
                                       value=20, step=1)
            data[f'SMA {period}'] = data['Adj Close'].rolling(period).mean()
            data2[f'SMA {period}'] = data[f'SMA {period}'].reindex(data2.index)

        sma2 = st.sidebar.checkbox('SMA2')
        if sma2:
            period2 = st.sidebar.slider('SMA2 period', min_value=5, max_value=500,
                                        value=100, step=1)
            data[f'SMA2 {period2}'] = data['Adj Close'].rolling(period2).mean()
            data2[f'SMA2 {period2}'] = data[f'SMA2 {period2}'].reindex(data2.index)

        st.subheader('Chart')
        st.line_chart(data2)

        if st.sidebar.checkbox('View statistic'):
            st.subheader('Statistic')
            st.table(data2.describe())

        if st.sidebar.checkbox('View quotes'):
            st.subheader(f'{asset} historical data')
            st.write(data2)

        if st.sidebar.checkbox('Predict closing price with Linear Regression'):
            model, X_test, y_test = create_model(data)
            y_pred = model.predict(X_test)

            # Convert ordinal X_test back to dates for visualization
            X_test_dates = [dt.datetime.fromordinal(x[0]) for x in X_test]

            # set seaborn style
            sns.set_style('darkgrid')

            # create plot
            fig, ax = plt.subplots(figsize=(10,5))
            ax.scatter(X_test_dates, y_test, color='gray', label='Actual price')
            ax.plot(X_test_dates, y_pred, color='red', linewidth=2, label='Predicted price')
            ax.set_title('Predicted vs Actual Closing Prices')
            ax.set_xlabel('Date')
            ax.set_ylabel('Closing Price')
            ax.legend()  # add legend
            plt.xticks(rotation=45)  # optional: rotate x-axis labels for better visibility

            st.pyplot(fig)

            st.caption("Note: This prediction is not an investment recommendation.")

if __name__ == '__main__':
    main()
