from operator import mod
import streamlit as st 
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
# %matplotlib inline
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib import style
from sklearn.metrics import accuracy_score
plt.style.use('dark_background')
st.title('Stock Price Predictor')


stocks = st.sidebar.selectbox(
    'Select Stock',
    ('Default','GOOG', 'BTC-USD', 'MSFT')
)
model_name = st.sidebar.selectbox(
        'Select Model',
        ('Default','Linear Regression', 'Lasso Regression', 'Ridge Regression')
    )
if stocks=='Default':
    st.write('Stock Predictor helps in detecting the closing price of stocks when the open , high , low values and volume for that particular is given as input..... though the feature is currently availabe for 3 stocks we will try to implement for more stocks in future. Happy trading... ')
else:
    st.write(f"## {stocks} Data")
    data=yf.download(stocks,"2015-01-01","2022-01-01",auto_adjust=True)
    # st.write(data)
    fig_open=plt.figure()
    plt.plot(data['Open'],linewidth=1)
    plt.title("Price Series")
    plt.ylabel("Opening Price")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    # plt.show()
    st.pyplot(fig_open)
    fig_open=plt.figure()
    sns.distplot(data['Open'])
    st.pyplot(fig_open)
    fig_high=plt.figure()
    plt.plot(data['High'],linewidth=1)
    plt.title("Price Series")
    plt.ylabel("Highest Price")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    # plt.show()
    st.pyplot(fig_high)
    fig_high=plt.figure()
    sns.distplot(data['High'])
    st.pyplot(fig_high)
    fig_low=plt.figure()
    plt.plot(data['Low'],linewidth=1)
    plt.title("Price Series")
    plt.ylabel("Lowest Price")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    # plt.show()
    st.pyplot(fig_low)
    fig_low=plt.figure()
    sns.distplot(data['Low'])
    st.pyplot(fig_low)
    fig_close=plt.figure()
    plt.plot(data['Close'],linewidth=1)
    plt.title("Price Series")
    plt.ylabel("Closing Price")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    # plt.show()
    st.pyplot(fig_close)
    fig_close=plt.figure()
    sns.distplot(data['Close'])
    st.pyplot(fig_close)
    fig_volume=plt.figure()
    plt.plot(data['Volume'],linewidth=1)
    plt.title("Price Series")
    plt.ylabel("Volume")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    # plt.show()
    st.pyplot(fig_volume)
    fig_volume=plt.figure()
    sns.distplot(data['Volume'])
    st.pyplot(fig_volume)

    x=data.drop('Close',axis=1)
    y=data['Close']



    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


    if model_name=='Linear Regression':
        from sklearn.linear_model import LinearRegression
        lr=LinearRegression()
    elif model_name=='Lasso Regression':
        from sklearn.linear_model import Lasso
        lr=Lasso()
    else:
        from sklearn.linear_model import Ridge
        lr=Ridge()


    lr.fit(x_train,y_train)
    Open = st.number_input('Open: ')
    High = st.number_input('High: ')
    Low = st.number_input('Low: ')
    Volume = st.number_input('Volume ')

    test_data = np.array([[Open,High,Low,Volume]])

    predictions=lr.predict(test_data)

    if st.button('Predict'):
        st.write("Prediction of closing price :-")
        st.write(predictions[0])


import webbrowser

url = 'https://www.linkedin.com/in/soham-chaudhuri-8aa0a9226/'

if st.button('LinkedIn'):
    webbrowser.open_new_tab(url)