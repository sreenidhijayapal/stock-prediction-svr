import csv
from array import array

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# initialise empty lists to store column values from csv
dates = []
prices = []


# function to fill the above list variables
def get_data(filename):
    with open(filename, 'r') as csvFile:
        # iterate over every row of the csv file
        csvFileReader = csv.reader(csvFile)
        # skip first row containing column names
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('/')[0]))
            prices.append(float(row[1].replace("$", "")))
    return


def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))

    # define 3 different kernels of the support vector regression models
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    # fit the data points in the models
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    # plot the initial datapoints
    plt.scatter(dates, prices, color='black', label='Data')
    # plot the line made by linear kernel
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    # plot the line made by polynomial kernel
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
    # plot the line made by the RBF kernel
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_lin.predict(x)[0], svr_poly.predict(x)[0], svr_rbf.predict(x)[0]


# call get_data method by passing the csv file to it
get_data("HistoricalData_1672751972899.csv")
# print "Dates- ", dates
# print "Prices- ", prices

predicted_price = predict_prices(dates, prices, 19)
