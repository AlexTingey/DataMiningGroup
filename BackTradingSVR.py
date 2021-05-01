from __future__ import (absolute_import, division, print_function,unicode_literals)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from pandas_datareader import data as pdr
from datetime import datetime
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import datetime
import os.path
import sys
import backtrader as bt
 
#Forward declaring helper method
def getBTCData(dateSpans):
  BTC_data = []
  for date in dateSpans:
    start_date = date[0]
    end_date = date[1]

    smolData = pdr.get_data_yahoo("BTC-USD", start=start_date, end=end_date)

    BTC_data.append(smolData)
  
  return pd.concat(BTC_data, axis=0)


"""
Method to predict closing date of next timeframe
Creates regression model using feature vectors of last 600 days

"""
def predictNextClose(date):
    #Get relevant feature vectors to train
    global prevValue
    indexList = list(btc_data.index.values)
    try:
        dateIdx = indexList.index(np.datetime64(date),0,len(indexList))
    except:
        return prevValue
    
    recDF = df[dateIdx-730:dateIdx]

    recBTC = btc_data['Close'][dateIdx-729:dateIdx+1]
    #Get relevant feature 
    scaler = MinMaxScaler()
    scale_data = scaler.fit_transform(recDF, recBTC)
    train_x, valid_x, train_y, valid_y = train_test_split(scale_data, recBTC, test_size = 0.2)


    svReg = SVR(C = 100000, epsilon = 0.5)
    #C represents tradeoff in minimizing the correctness of the classifier and allowing support vectors
    # epsilon is our error tolerance
    svReg.fit(train_x,train_y)
    x_pred = svReg.predict(valid_x)
    prevValue = x_pred[-1]
    return x_pred[-1]

class SmaCross(bt.SignalStrategy):
    def __init__(self):
        sma1, sma2 = bt.ind.SMA(period=10), bt.ind.SMA(period=30)
        crossover = bt.ind.CrossOver(sma1, sma2)
        self.signal_add(bt.SIGNAL_LONG, crossover)

class MyAllInSizer(bt.Sizer):
    def _getsizing(self,comminfo, cash, data, isbuy):
        if isbuy:
            size = self.broker.get_cash() / self.strategy.datas[0].open
            return size
        position = self.broker.getposition(data)
        if not position.size:
            return 0
        size = self.broker.getposition(data = self.strategy.datas[0]).size
        return size

class SVR_SMA(bt.SignalStrategy):
    def log(self, txt, dt = None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def next(self):
        pred = predictNextClose(self.datas[0].datetime.date(0))
        self.log('Close, %.2f' % self.dataclose[0])

        #We're already trying to buy or sell - Must wait
        if self.order:
            return

        #if we aren't in the market, could be time to buy
        if not self.position:
            if pred >self.dataclose[0]:
                #If we're predicting price to go up, after it's been going down, we buy
                if self.dataclose[0] < self.dataclose[-1]:
                    self.log('YOLO $WAG, %.2f' % self.dataclose[0])
                    self.order = self.buy()
        #If we're in the market time to see if we sell
        else:
            if pred < self.dataclose[0]:
                if self.dataclose[0] > self.dataclose[-1]:#If prices projected to dip after they've been rising, we sell
                    if self.dataclose[-1] > self.dataclose[-2]:
                       self.log('PAPER HANDED BITCH, %.2f' % self.dataclose[0])
                       self.order = self.sell()


"""
#Import our tokenizer and train our model for nlp sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")						
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

#Train examples for our nlp
good_example = "I love big bubbles! This is the biggest bubble I have ever seen!"
bad_example = "Bearish indicators show that a crash could be imminent"
normal_example = "Money is good"

#Finish setting up model
good_example_tokenized = tokenizer.encode_plus(good_example, return_tensors="pt")
classification_logits = model(**good_example_tokenized)[0]
classes = ["positive", "negative", "neutral"]
evaluate = torch.softmax(classification_logits, dim=1)[0]
"""
dataSpans = [('2014-09-19', '2016-01-01'), ('2016-02-02', '2016-08-01'), ('2016-09-02', '2016-12-01'), 
('2017-02-02', '2017-02-25'), ('2017-05-02', '2017-07-01'), ('2017-08-02', '2017-09-01'), 
('2018-01-02', '2020-01-01')]
btc_data = getBTCData(dataSpans)
df = pd.read_csv('C:\\Users\\Spencer\\Desktop\\DataMining\\DataMiningGroup\\stackedData.csv')
df = df.drop(['Unnamed: 0', '0'], axis=1)
data_copy = df[:]
data_copy['BTC_OPEN'] = btc_data['Open'].values
data_copy['BTC_HIGH'] = btc_data['High'].values
data_copy['BTC_LOW'] = btc_data['Low'].values
data_copy['BTC_VOLUME'] = btc_data['Volume'].values
df= data_copy
prevValue = 0

#Trying to do the backtracker here
cerebro = bt.Cerebro()
data = bt.feeds.YahooFinanceData(dataname='BTC-USD', fromdate=datetime.datetime(2019, 1, 1), todate=datetime.datetime(2020, 1, 1))
cerebro.adddata(data)
#cerebro.addsizer(MyAllInSizer)
cerebro.broker.set_cash(10000)
cerebro.addstrategy(SVR_SMA)

# Print out the starting conditions
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Run over everything
cerebro.run()

# Print out the final result
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

