from flask import Flask,request
from finvizfinance.screener.overview import Overview
from flask_jsonpify import jsonpify
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import datetime as dt
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
app = Flask(__name__)

@app.route("/filter")
def index():
    foverview = Overview()
    filters_dict = {'EPS growthqtr over qtr':'Positive (>0%)','P/E':' '}
    query_1=request.args.get('name_1')
    query_2=request.args.get('name_2')
    query={'EPS growthqtr over qtr':str(query_1),'P/E':str(query_2)}
    foverview.set_filter(filters_dict=query)
    df = foverview.screener_view()
    df=df.values.tolist()
    json=jsonpify(df)
    return json
@app.route('/predict')
def predict():
    symbol=request.args.get('sys')
    sp500 = yf.Ticker(str(symbol))
    sp500 = sp500.history(period="max")
    sp500.index = pd.to_datetime(sp500.index)
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    sp500 = sp500.loc["1990-01-01":].copy()
    predictors = ["Close", "Volume", "Open", "High", "Low"]
    horizons = [2,5,60,250,1000]
    for horizon in horizons:
        rolling_averages = sp500.rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
        trend_column = f"Trend_{horizon}"
        sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
        predictors+= [ratio_column, trend_column]

    sp500 = sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"])
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
    model.fit(sp500[predictors],sp500['Target'])
    res=model.predict(sp500.tail(1)[predictors])
    if res==0:
        return jsonpify(prediction='Price may fall down')
    else:
        return jsonpify(prediction='Price may increase')
def load_data(ticker,START,TODAY):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data
def predict(index,symbol):
    START = "2018-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    data = load_data(symbol,START,TODAY)
    train = pd.DataFrame(data[0:int(len(data)*0.70)])
    test = pd.DataFrame(data[int(len(data)*0.70): int(len(data))])
    scaler = MinMaxScaler(feature_range=(0,1))
    train_close = train.iloc[:, index:index+1].values
    test_close = test.iloc[:, index:index+1].values
    data_training_array = scaler.fit_transform(train_close)
    x_train = []
    y_train = [] 
    for i in range(10, data_training_array.shape[0]):
        x_train.append(data_training_array[i-10: i])
        y_train.append(data_training_array[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    model = Sequential()
    model.add(LSTM(units = 50, activation = 'relu', return_sequences=True
              ,input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 60, activation = 'relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units = 80, activation = 'relu', return_sequences=False))
    model.add(Dense(units = 1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    model.fit(x_train, y_train,epochs = 1)
    past_100_days = pd.DataFrame(train_close[-10:])
    test_df = pd.DataFrame(test_close)
    final_df = past_100_days.append(test_df, ignore_index = True)
    input_data = scaler.fit_transform(final_df)
    x_test = []
    y_test = []
    for i in range(10, input_data.shape[0]+1):
        x_test.append(input_data[i-10: i])
    for i in range(10, input_data.shape[0]):
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_pred = model.predict(x_test)
    predicted=scaler.inverse_transform(y_pred)
    y_test=y_test.reshape(-1,1)
    nor_y=scaler.inverse_transform(y_test)
    return predicted[-1],predicted[-2]

@app.route('/open')
def predict_open():
    symbol=request.args.get('sys')
    tom,today=predict(1,symbol)
    
    return jsonpify(prediction='Tomarrow opening price is {}'.format(tom),today='Today predicted opening price is {}'.foramt(today))
@app.route('/close')
def predict_close():
    symbol=request.args.get('sys')
    tom,today=predict(4,symbol)
    
    return jsonpify(prediction='Tommarrow closing  price is {}'.format(tom),today='Today predicted closing price is {}'.foramt(today))
@app.route('/High')
def predict_high():
    symbol=request.args.get('sys')
    tom,today=predict(2,symbol)
    return jsonpify(prediction='Tomarrow High  price is {}'.format(tom),today='Today predicted Hight price is {}'.foramt(today))
@app.route('/low')
def predict_low():
    symbol=request.args.get('sys')
    tom,today=predict(3,symbol)
    return jsonpify(prediction='Tomarrow Low price is {}'.format(tom),today='Today predicted low price is {}'.foramt(today))


