# We use a Python script that uses the Streamlit framework to create an interactive web application for visualizing and analyzing historical stock data from Yahoo Finance. It includes functions to display stock charts, user defined moving averages (simple moving averages – SMA, SMA2), statistics, and perform linear regression on user defined timeframes to predict closing prices.

# First Importing necessary libraries:

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

# These libraries provide functions for data manipulation, machine learning, visualization, and Streamlit app creation.

@st.cache
def load_data():
    components = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return components.set_index('Symbol')
    
# The @st.cache decorator is used to cache the output of a function. It helps improve the performance of the Streamlit app by preventing unnecessary repeated calculations when the same function is called with the same arguments.
# load_data() function: loads data about the components of the S&P 500 from a Wikipedia page and returns a DataFrame in which the symbols are set as an index. 

def load_quotes(asset):
    return yf.download(asset)

# Downloads historical stock data using the yfinance library for a specific stock symbol (asset).

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

# The function prepares the data for linear regression. It converts the date index into ordinal values and then performs linear regression to predict the closing prices (close) based on the data.

def main():
    components = load_data()
    title = st.empty()

# This is the main function of the Streamlit app. It loads the data of the S&P 500 components, initializes the Streamlit App interface and sets up the main elements of the user interface.

    def label(symbol):
        return symbol

    asset = st.sidebar.selectbox('Select asset', components.index.sort_values(), index=3, format_func=label)
    title.title(asset)

# These lines create a Streamlit sidebar with a dropdown selection box to select an asset. Then, the title of the main interface is set to the selected asset.

    data0 = load_quotes(asset)
    if data0 is not None:
        data = data0.copy().dropna()
        data.index.name = None

# In this section, the load_quotes() function loads historical stock data for the selected asset. Then a copy of the data is created, all rows with missing values are deleted and the index name is removed.

# Visualization and interaction options:
        
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

# This part of the code allows the user to interact with the application by selecting the number of quotes to display, enabling/disabling simple moving averages (SMA), and adjusting the time periods for SMA calculations.

        st.subheader('Chart')
        st.line_chart(data2)

# This displays a line chart using the st.line_chart() function to visualize the historical closing prices and optionally the selected simple moving averages.

        if st.sidebar.checkbox('View statistic'):
            st.subheader('Statistic')
            st.table(data2.describe())

        if st.sidebar.checkbox('View quotes'):
            st.subheader(f'{asset} historical data')
            st.write(data2)

# In this section you will find options to view statistics and historical data for the selected asset.

        if st.sidebar.checkbox('Predict closing price with Linear Regression'):
            model, X_test, y_test = create_model(data)
            y_pred = model.predict(X_test)

            # Convert ordinal X_test back to dates for visualization
            X_test_dates = [dt.datetime.fromordinal(x[0]) for x in X_test]

            # Set seaborn style
            sns.set_style('darkgrid')

            # Create plot
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

# In this section, users can predict closing prices using linear regression and plot the predicted versus actual closing prices on a chart. It also displays a note in the caption that the predictions of the model are not investment recommendations.

if __name__ == '__main__':
    main()
    
# This line ensures that the main() function is executed when the script is run directly (not imported as a module).

# In summary, the code creates an interactive web application that uses Streamlit to examine historical stock data, visualize price trends, calculate moving averages, and perform basic linear regression predictions. It provides a user-friendly interface for interacting with and analyzing stock data.
