# |___________________________|
# | Linear Regression Example |
# | Author: LeMi11ion         |
# | Cred: sentdex, YT Channel |
# |___________________________|

import pandas as pd
import quandl, math, datetime, sys
import numpy as np
from enum import Enum
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use("ggplot")
quandl.ApiConfig.api_key = "ta-dTwRunWYT48XLb1gC"

class AlgorithmType(Enum):
    linear = "linear"
    polynomial = "polynomial"

class ArgumentList(Enum):
    algorithm = "algorithm"
    train = "train"

#default algorithm type and load
algorithmType = AlgorithmType.linear
train = False

#get cli arguments
argLen = len(sys.argv)
for arg in sys.argv:
    if (sys.argv.index(arg) == 0):
        continue

    args = arg.split(":")
    if (len(args) != 2):
        raise Exception("Argument "+arg+" is not of the right format. Must be 'type:value'")

    prefix = args[0]
    suffix = args[1]
    delimiter = ","

    if (prefix == ArgumentList.algorithm.value):
        if (suffix == AlgorithmType.polynomial.value):
            algorithmType = AlgorithmType.polynomial
        elif (suffix == AlgorithmType.linear.value):
            algorithmType = AlgorithmType.linear
        else:
            errorMsg = "Param for "+ArgumentList.algorithm.value+" must be:\n"
            listType = "["
            listType += AlgorithmType.linear.value + delimiter
            listType += AlgorithmType.polynomial.value + "]"
            raise Exception(errorMsg+listType)
    elif (prefix == ArgumentList.train.value):
        if (suffix == "True"):
            train = True
        elif (suffix == "False"):
            trian = False
        else:
            errorMsg = "Param for "+ArgumentList.train.value+" must be:\n"
            listType = "["
            listType += "True" + delimiter
            listType += "False" + "]"
            raise Exception(errorMsg+listType)
    else:
        errorMsg = "Argument "+prefix+" is not recognized. Must be one of the following:\n"
        listType = "["
        listType += ArgumentList.algorithm.value + delimiter
        listType += ArgumentList.train.value + "]"
        raise Exception(errorMsg+listType)

#get the data set from quandl
df = quandl.get("WIKI/GOOGL")

#cut the data down to what we want to use
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume",]]

#develop our features
df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"] + 100.0
df["PCT_change"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100.0
df = df[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]

#we want to forecast the close of the stock
forecast_col = "Adj. Close"
#quandl gives Nan, convert to outliers
df.fillna(-99999, inplace=True)

#We want to frocast 1% of the data set, 35 days
forecast_out = int(math.ceil(0.01*len(df)))

#shift the data set to not include that one percent
df["label"] = df[forecast_col].shift(-forecast_out)

#filter out labels
x = np.array(df.drop(["label"],1))
x = preprocessing.scale(x)
#future data
x_lately = x[-forecast_out:]
#training set
x = x[:-forecast_out]

df.dropna(inplace=True)

#the closing labels
y = np.array(df["label"])

#splits data randomly into training and testing sets (test size is 20%)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

if (train):
    #get linear regression algorithm object
    if (algorithmType == AlgorithmType.linear):
        print("Retraining model with linear regression...")
        clf = LinearRegression(n_jobs=-1)
        clf.fit(x_train, y_train)
    else:
        print("Retraining model with polynomial regression...")
        clf = make_pipeline(preprocessing.PolynomialFeatures(degree=2), LinearRegression())
        clf.fit(x_train, y_train)

    #save the algorithm
    with open("linear_regression.pickle","wb") as f:
        pickle.dump(clf, f)

else:
    print("Loading file linear_regression.pickle")
    #load in the algorithm for use
    pickle_in = open("linear_regression.pickle", "rb")
    clf = pickle.load(pickle_in)

accuracy = clf.score(x_test, y_test)

#using the future data predict using the algorithm
forecast_set = clf.predict(x_lately)
print(accuracy)
#set the forecast column to be 
df["Forecast"] = np.nan

#get the last item name in the df 
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
#number of seconds in a day
one_day = 86400
next_unix = last_unix + one_day

#loop through the forecast set
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    #set the next item in the data to be a row full of NaN except for the column
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df["Adj. Close"].plot()
df["Forecast"].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
