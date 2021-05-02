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
Method to predict closing date of next timeframe nadv in future
Creates regression model using feature vectors of last 2 yrs
"""
def predictFuture(date,nAdv):
    #Get relevant feature vectors to train
    global prevValue
    indexList = list(btc_data.index.values)
    try:
        dateIdx = indexList.index(np.datetime64(date),0,len(indexList))
    except:
        return prevValue
    
    recDF = df[dateIdx-730:dateIdx]

    recBTC = btc_data['Close'][dateIdx-(730-nAdv):(dateIdx + nAdv)]
    
    #Get relevant feature 
    scaler = MinMaxScaler()
    scale_data = scaler.fit_transform(recDF, recBTC)
    train_x, valid_x, train_y, valid_y = train_test_split(scale_data, recBTC, test_size = 0.2)
    svReg = SVR(C = 100000, epsilon = 0.5)
    #C represents tradeoff in minimizing the correctness of the classifier and allowing support vectors
    # epsilon is our error tolerance
    svReg.fit(train_x,train_y)
    x_pred = svReg.predict(recDF)
    prevValue = x_pred[-1]
    return x_pred[-1]

#Run on init, want to predict sma of startdate+1,startdate+2,...,startdate+n nums
def initNDaySMA(date,n):
    delta = datetime.timedelta(days = 1)
    arr = []
    for i in range(1,n+1):
        arr.append(predictFuture(date,i))
    return arr

#Signal 0 means hold, 1 means buy, -1 means sell
def smaSignal(sma1,sma2,prevSma1i,prevSma2i):
    global prevSma1
    global prevSma2
    global sma1Size
    global sma2Size
    ret = 0
    #We look for sma1 to cross sma2 w/ positive slope -Indicates Buy
    if(sum(prevSma1i)/sma1Size < sum(prevSma2i)/sma2Size):
        if(sum(sma1)/sma1Size >sum(sma2)/sma2Size):
            ret = 1
    #We look for sma1 to cross sma2 w/negative slope -Indicates Sell
    else:
        if (sum(sma1)/sma1Size<sum(sma2)/sma2Size):
            ret =  -1

    prevSma1 = list.copy(sma1)
    prevSma2 = list.copy(sma2)
    return ret


class SmaCross(bt.SignalStrategy):
    def __init__(self):
        sma1, sma2 = bt.ind.SMA(period=10), bt.ind.SMA(period=30)
        crossover = bt.ind.CrossOver(sma1, sma2)
        self.signal_add(bt.SIGNAL_LONG, crossover)

class MyAllInSizer(bt.Sizer):
    def _getsizing(self,comminfo, cash, data, isbuy):
        if isbuy:
            size = int(self.broker.get_cash() / self.strategy.datas[0].open)
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
        self.sma1 = [0]
        self.sma2 = [0]

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
        self.log('Close, %.2f' % self.dataclose[0])
        if(len(self.sma1) <10):#means this is first iter
            self.sma1 = initNDaySMA(self.datas[0].datetime.date(0),sma1Size)
        else:
            self.sma1.pop(0)
            self.sma1.append(predictFuture(self.datas[0].datetime.date(0),sma1Size))

        if(len(self.sma2) <20):#means this is first iter
            self.sma2 = initNDaySMA(self.datas[0].datetime.date(0),sma2Size)
        else:
            self.sma2.pop(0)
            self.sma2.append(predictFuture(self.datas[0].datetime.date(0),sma2Size))
        indicator = smaSignal(self.sma1,self.sma2,prevSma1,prevSma2)
        #Indicator 1 means buy
        if indicator == 1:
            self.log('YOLO $WAG, %.2f' % self.dataclose[0])
            self.order = self.buy()
        #-1 means sell
        if self.position:#only try to sell if in market
            if indicator == -1:
                self.log('Selling now, %.2f' % self.dataclose[0])
                self.order = self.sell()
        #If we reached this point, neither buy nor sell
        

class SVR_shitbrainstrat(bt.Strategy):
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
        pred = predictFuture(self.datas[0].datetime.date(0),1)
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
                       self.log('Selling, %.2f' % self.dataclose[0])
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
numTrials = 100
finalPortfolioValues = []
for i in range(numTrials):
    dataSpans = [('2014-09-19', '2016-01-01'), ('2016-02-02', '2016-08-01'), ('2016-09-02', '2016-12-01'), 
    ('2017-02-02', '2017-02-25'), ('2017-05-02', '2017-07-01'), ('2017-08-02', '2017-09-01'), 
    ('2018-01-02', '2020-01-01')]
    btc_data = getBTCData(dataSpans)
    df = pd.read_csv('C:\\Users\\Alex\\Desktop\\Spring 2021\\Data Mining\\Group project\\dataFiles\\stackedData.csv')
    df = df.drop(['Unnamed: 0', '0'], axis=1)
    data_copy = df[:]
    data_copy['BTC_OPEN'] = btc_data['Open'].values
    data_copy['BTC_HIGH'] = btc_data['High'].values
    data_copy['BTC_LOW'] = btc_data['Low'].values
    data_copy['BTC_VOLUME'] = btc_data['Volume'].values
    df= data_copy
    prevValue = 0
    fda = 0
    tda = 0
    #Trying to do the backtracker here
    cerebro = bt.Cerebro()
    data = bt.feeds.YahooFinanceData(dataname='BTC-USD', fromdate=datetime.datetime(2019, 1, 1), todate=datetime.datetime(2019, 12, 15))
    cerebro.adddata(data)
    cerebro.addsizer(MyAllInSizer)
    cerebro.broker.set_cash(10000)
    #cerebro.addstrategy(SVR)
    sma1Size = 10
    sma2Size = 20
    prevSma1 = [0]*sma1Size
    prevSma2 = [0]*sma2Size
    cerebro.addstrategy(SVR_SMA)
    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    finalPortfolioValues.append(cerebro.broker.getvalue())
print(finalPortfolioValues)
pct_change = [((x - 10,000)/(10,000)) for x in finalPortfolioValues]
print(f'Average increase: {sum(pct_change)/len(pct_change)}')