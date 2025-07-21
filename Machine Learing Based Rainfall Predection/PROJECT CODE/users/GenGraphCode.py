import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
import os
import matplotlib
matplotlib.use('TKAgg')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plot_graphs(groundtruth, prediction, title):
    N = 9
    ind = np.arange(N)  # the x locations for the groups
    width = 0.27  # the width of the bars

    fig = plt.figure()
    fig.suptitle(title, fontsize=12)
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, groundtruth, width, color='r')
    rects2 = ax.bar(ind + width, prediction, width, color='g')

    ax.set_ylabel("Amount of rainfall")
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'))
    ax.legend((rects1[0], rects2[0]), ('Ground truth', 'Prediction'))

    #     autolabel(rects1)
    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * h, '%d' % int(h),
                ha='center', va='bottom')
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * h, '%d' % int(h),
                ha='center', va='bottom')
    #     autolabel(rects2)
    # os.path.join(BASE_DIR, 'assets/templates')
    plt.savefig(os.path.join(BASE_DIR, 'assets/static/graphs/') + title + "." + 'png')
    plt.show()
    plt.close()


class GeneratePltGraph:

    def preProcessGraphs(self,data):
        data = data.fillna(data.mean())
        data.info()
        print(data.head())
        print(data.describe())
        # plt.savefig('models_accuracy.png') #for Saving pic
        # plt.savefig('figure1.png',data.hist(figsize=(24, 24)))
        data.hist(figsize=(24, 24));
        #plt.savefig('hist.png')
        data.groupby("YEAR").sum()['ANNUAL'].plot(figsize=(12, 8));
        data[['YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby(
            "YEAR").sum().plot(figsize=(13, 8));
        data[['YEAR', 'JanToFeb', 'MarToMay', 'JunToSep', 'OctToDec']].groupby("YEAR").sum().plot(figsize=(13, 8));
        data[['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV',
              'DEC']].groupby("SUBDIVISION").mean().plot.barh(stacked=True, figsize=(13, 8));
        data[['SUBDIVISION', 'JanToFeb', 'MarToMay', 'JunToSep', 'OctToDec']].groupby("SUBDIVISION").sum().plot.barh(
            stacked=True, figsize=(16, 8));
        plt.figure(figsize=(11, 4))
        sns.heatmap(data[['JanToFeb', 'MarToMay', 'JunToSep', 'OctToDec', 'ANNUAL']].corr(), annot=True)
        plt.show()

        plt.figure(figsize=(11, 4))
        sns.heatmap(
            data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANNUAL']].corr(),
            annot=True)
        plt.show()
        plt.close()



    def genMlrCodes(self,data):
        # seperation of training and testing data


        division_data = np.asarray(
            data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']])

        X = None;
        y = None
        for i in range(division_data.shape[1] - 3):
            if X is None:
                X = division_data[:, i:i + 3]
                y = division_data[:, i + 3]
            else:
                X = np.concatenate((X, division_data[:, i:i + 3]), axis=0)
                y = np.concatenate((y, division_data[:, i + 3]), axis=0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        # test 2010
        temp = data[['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
                     'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[data['YEAR'] == 2010]

        data_2010 = np.asarray(temp[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
                                     'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[temp['SUBDIVISION'] == 'TELANGANA'])

        X_year_2010 = None;
        y_year_2010 = None
        for i in range(data_2010.shape[1] - 3):
            if X_year_2010 is None:
                X_year_2010 = data_2010[:, i:i + 3]
                y_year_2010 = data_2010[:, i + 3]
            else:
                X_year_2010 = np.concatenate((X_year_2010, data_2010[:, i:i + 3]), axis=0)
                y_year_2010 = np.concatenate((y_year_2010, data_2010[:, i + 3]), axis=0)

        # test 2005
        temp = data[['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
                     'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[data['YEAR'] == 2005]

        data_2005 = np.asarray(temp[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
                                     'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[temp['SUBDIVISION'] == 'TELANGANA'])

        X_year_2005 = None;
        y_year_2005 = None
        for i in range(data_2005.shape[1] - 3):
            if X_year_2005 is None:
                X_year_2005 = data_2005[:, i:i + 3]
                y_year_2005 = data_2005[:, i + 3]
            else:
                X_year_2005 = np.concatenate((X_year_2005, data_2005[:, i:i + 3]), axis=0)
                y_year_2005 = np.concatenate((y_year_2005, data_2005[:, i + 3]), axis=0)
        # terst 2015
        temp = data[['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
                     'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[data['YEAR'] == 2015]

        data_2015 = np.asarray(temp[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
                                     'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[temp['SUBDIVISION'] == 'TELANGANA'])

        X_year_2015 = None;
        y_year_2015 = None
        for i in range(data_2015.shape[1] - 3):
            if X_year_2015 is None:
                X_year_2015 = data_2015[:, i:i + 3]
                y_year_2015 = data_2015[:, i + 3]
            else:
                X_year_2015 = np.concatenate((X_year_2015, data_2015[:, i:i + 3]), axis=0)
                y_year_2015 = np.concatenate((y_year_2015, data_2015[:, i + 3]), axis=0)

        from sklearn import linear_model

        # linear model
        reg = linear_model.ElasticNet(alpha=0.5)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print(mean_absolute_error(y_test, y_pred))

        y_year_pred_2005 = reg.predict(X_year_2005)

        # 2010
        y_year_pred_2010 = reg.predict(X_year_2010)

        y_year_pred_2015 = reg.predict(X_year_2015)

        print("MEAN 2005")
        print(np.mean(y_year_2005), np.mean(y_year_pred_2005))
        print("Standard deviation 2005")
        print(np.sqrt(np.var(y_year_2005)), np.sqrt(np.var(y_year_pred_2005)))

        print("MEAN 2010")
        print(np.mean(y_year_2010), np.mean(y_year_pred_2010))
        print("Standard deviation 2010")
        print(np.sqrt(np.var(y_year_2010)), np.sqrt(np.var(y_year_pred_2010)))

        print("MEAN 2015")
        print(np.mean(y_year_2015), np.mean(y_year_pred_2015))
        print("Standard deviation 2015")
        print(np.sqrt(np.var(y_year_2015)), np.sqrt(np.var(y_year_pred_2015)))

        plot_graphs(y_year_2005, y_year_pred_2005, "Year-2005")
        plot_graphs(y_year_2010, y_year_pred_2010, "Year-2010")
        plot_graphs(y_year_2015, y_year_pred_2015, "Year-2015")
        plt.close()

        """from sklearn.svm import SVR

        # SVM model
        clf = SVR(gamma='auto', C=0.1, epsilon=0.2)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(mean_absolute_error(y_test, y_pred))
        # 2005
        y_year_pred_2005 = reg.predict(X_year_2005)

        # 2010
        y_year_pred_2010 = reg.predict(X_year_2010)

        # 2015
        y_year_pred_2015 = reg.predict(X_year_2015)
        print("MEAN 2005")
        print(np.mean(y_year_2005), np.mean(y_year_pred_2005))
        print("Standard deviation 2005")
        print(np.sqrt(np.var(y_year_2005)), np.sqrt(np.var(y_year_pred_2005)))

        print("MEAN 2010")
        print(np.mean(y_year_2010), np.mean(y_year_pred_2010))
        print("Standard deviation 2010")
        print(np.sqrt(np.var(y_year_2010)), np.sqrt(np.var(y_year_pred_2010)))

        print("MEAN 2015")
        print(np.mean(y_year_2015), np.mean(y_year_pred_2015))
        print("Standard deviation 2015")
        print(np.sqrt(np.var(y_year_2015)), np.sqrt(np.var(y_year_pred_2015)))

        plot_graphs(y_year_2005, y_year_pred_2005, "Year-2005")
        plot_graphs(y_year_2010, y_year_pred_2010, "Year-2010")
        plot_graphs(y_year_2015, y_year_pred_2015, "Year-2015")

        from keras.models import Model
        from keras.layers import Dense, Input, Conv1D, Flatten

        # NN model
        inputs = Input(shape=(3, 1))
        x = Conv1D(64, 2, padding='same', activation='elu')(inputs)
        x = Conv1D(128, 2, padding='same', activation='elu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='elu')(x)
        x = Dense(64, activation='elu')(x)
        x = Dense(32, activation='elu')(x)
        x = Dense(1, activation='linear')(x)
        model = Model(inputs=[inputs], outputs=[x])
        model.compile(loss='mean_squared_error', optimizer='adamax', metrics=['mae'])
        model.summary()

        model.fit(x=np.expand_dims(X_train, axis=2), y=y_train, batch_size=64, epochs=10, verbose=1, validation_split=0.1,
                  shuffle=True)
        y_pred = model.predict(np.expand_dims(X_test, axis=2))
        print(mean_absolute_error(y_test, y_pred))
        # 2005
        y_year_pred_2005 = reg.predict(X_year_2005)

        # 2010
        y_year_pred_2010 = reg.predict(X_year_2010)

        # 2015
        y_year_pred_2015 = reg.predict(X_year_2015)

        print("MEAN 2005")
        print(np.mean(y_year_2005), np.mean(y_year_pred_2005))
        print("Standard deviation 2005")
        print(np.sqrt(np.var(y_year_2005)), np.sqrt(np.var(y_year_pred_2005)))

        print("MEAN 2010")
        print(np.mean(y_year_2010), np.mean(y_year_pred_2010))
        print("Standard deviation 2010")
        print(np.sqrt(np.var(y_year_2010)), np.sqrt(np.var(y_year_pred_2010)))

        print("MEAN 2015")
        print(np.mean(y_year_2015), np.mean(y_year_pred_2015))
        print("Standard deviation 2015")
        print(np.sqrt(np.var(y_year_2015)), np.sqrt(np.var(y_year_pred_2015)))

        plot_graphs(y_year_2005, y_year_pred_2005, "Year-2005")
        plot_graphs(y_year_2010, y_year_pred_2010, "Year-2010")
        plot_graphs(y_year_2015, y_year_pred_2015, "Year-2015")

        # spliting training and testing data only for telangana
        telangana = np.asarray(data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
                                     'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[data['SUBDIVISION'] == 'TELANGANA'])

        X = None;
        y = None
        for i in range(telangana.shape[1] - 3):
            if X is None:
                X = telangana[:, i:i + 3]
                y = telangana[:, i + 3]
            else:
                X = np.concatenate((X, telangana[:, i:i + 3]), axis=0)
                y = np.concatenate((y, telangana[:, i + 3]), axis=0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

        from sklearn import linear_model

        # linear model
        reg = linear_model.ElasticNet(alpha=0.5)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print(mean_absolute_error(y_test, y_pred))

        y_year_pred_2005 = reg.predict(X_year_2005)

        # 2010
        y_year_pred_2010 = reg.predict(X_year_2010)

        # 2015
        y_year_pred_2015 = reg.predict(X_year_2015)

        print("MEAN 2005")
        print(np.mean(y_year_2005), np.mean(y_year_pred_2005))
        print("Standard deviation 2005")
        print(np.sqrt(np.var(y_year_2005)), np.sqrt(np.var(y_year_pred_2005)))

        print("MEAN 2010")
        print(np.mean(y_year_2010), np.mean(y_year_pred_2010))
        print("Standard deviation 2010")
        print(np.sqrt(np.var(y_year_2010)), np.sqrt(np.var(y_year_pred_2010)))

        print("MEAN 2015")
        print(np.mean(y_year_2015), np.mean(y_year_pred_2015))
        print("Standard deviation 2015")
        print(np.sqrt(np.var(y_year_2015)), np.sqrt(np.var(y_year_pred_2015)))

        plot_graphs(y_year_2005, y_year_pred_2005, "Year-2005")
        plot_graphs(y_year_2010, y_year_pred_2010, "Year-2010")
        plot_graphs(y_year_2015, y_year_pred_2015, "Year-2015")

        from sklearn.svm import SVR

        # SVM model
        clf = SVR(kernel='rbf', gamma='auto', C=0.5, epsilon=0.2)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(mean_absolute_error(y_test, y_pred))

        # 2005
        y_year_pred_2005 = reg.predict(X_year_2005)

        # 2010
        y_year_pred_2010 = reg.predict(X_year_2010)

        # 2015
        y_year_pred_2015 = reg.predict(X_year_2015)

        print("MEAN 2005")
        print(np.mean(y_year_2005), np.mean(y_year_pred_2005))
        print("Standard deviation 2005")
        print(np.sqrt(np.var(y_year_2005)), np.sqrt(np.var(y_year_pred_2005)))

        print("MEAN 2010")
        print(np.mean(y_year_2010), np.mean(y_year_pred_2010))
        print("Standard deviation 2010")
        print(np.sqrt(np.var(y_year_2010)), np.sqrt(np.var(y_year_pred_2010)))

        print("MEAN 2015")
        print(np.mean(y_year_2015), np.mean(y_year_pred_2015))
        print("Standard deviation 2015")
        print(np.sqrt(np.var(y_year_2015)), np.sqrt(np.var(y_year_pred_2015)))

        plot_graphs(y_year_2005, y_year_pred_2005, "Year-2005")
        plot_graphs(y_year_2010, y_year_pred_2010, "Year-2010")
        plot_graphs(y_year_2015, y_year_pred_2015, "Year-2015")

        model.fit(x=np.expand_dims(X_train, axis=2), y=y_train, batch_size=64, epochs=10, verbose=1, validation_split=0.1,
                  shuffle=True)
        y_pred = model.predict(np.expand_dims(X_test, axis=2))
        print(mean_absolute_error(y_test, y_pred))

        # 2005
        y_year_pred_2005 = reg.predict(X_year_2005)

        # 2010
        y_year_pred_2010 = reg.predict(X_year_2010)

        # 2015
        y_year_pred_2015 = reg.predict(X_year_2015)

        print("MEAN 2005")
        print(np.mean(y_year_2005), np.mean(y_year_pred_2005))
        print("Standard deviation 2005")
        print(np.sqrt(np.var(y_year_2005)), np.sqrt(np.var(y_year_pred_2005)))

        print("MEAN 2010")
        print(np.mean(y_year_2010), np.mean(y_year_pred_2010))
        print("Standard deviation 2010")
        print(np.sqrt(np.var(y_year_2010)), np.sqrt(np.var(y_year_pred_2010)))

        print("MEAN 2015")
        print(np.mean(y_year_2015), np.mean(y_year_pred_2015))
        print("Standard deviation 2015")
        print(np.sqrt(np.var(y_year_2015)), np.sqrt(np.var(y_year_pred_2015)))

        plot_graphs(y_year_2005, y_year_pred_2005, "Year-2005")
        plot_graphs(y_year_2010, y_year_pred_2010, "Year-2010")
        plot_graphs(y_year_2015, y_year_pred_2015, "Year-2015")"""


    def testMltMSE(self,data):
        rsltdict = {}
        X = data['JAN'].values
        y = data['DEC'].values
        X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        # print('Predection result ',y_pred)
        # print('Test ',X_test)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        lrcorr = regressor.coef_
        lcoe = sum(lrcorr) / len(lrcorr)
        rsltdict.update({'lgmse': mse, 'lgrmse': rmse, 'lgcorr': lcoe})
        accuracy = metrics.accuracy_score(y_test.round(), y_pred.round(), normalize=False)
        # print('LR accuracy ', accuracy)

        # print(mse, "==", rmse)

        division_data = np.asarray(data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
                                         'AUG', 'SEP', 'OCT', 'NOV', 'DEC']])

        X = None;
        y = None
        for i in range(division_data.shape[1] - 3):
            if X is None:
                X = division_data[:, i:i + 3]
                y = division_data[:, i + 3]
            else:
                X = np.concatenate((X, division_data[:, i:i + 3]), axis=0)
                y = np.concatenate((y, division_data[:, i + 3]), axis=0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        # print('Predection result ',y_pred)
        # print('Test ',X_test)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mlrco = regressor.coef_
        mlrcorre = sum(mlrco) / len(mlrco)
        # accuracy = metrics.accuracy_score(y_test.round(), y_pred.round(),normalize=False)
        # print('MLR accuracy ',accuracy)

        print(mse, "==", rmse)
        # plt.scatter(x=y_test, y=y_pred, c="red")
        # plt.show()
        rsltdict.update({'mlrmse': mse, 'mlrrmse': rmse, 'mlrcorr': lcoe})

        ####QPF COde
        '''from keras.models import Model
        from keras.layers import Dense, Input, Conv1D, Flatten

        # NN model
        inputs = Input(shape=(3, 1))
        x = Conv1D(64, 2, padding='same', activation='elu')(inputs)
        x = Conv1D(128, 2, padding='same', activation='elu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='elu')(x)
        x = Dense(64, activation='elu')(x)
        x = Dense(32, activation='elu')(x)
        x = Dense(1, activation='linear')(x)
        model = Model(inputs=[inputs], outputs=[x])
        model.compile(loss='mean_squared_error', optimizer='adamax', metrics=['mae'])
        model.summary()

        model.fit(x=np.expand_dims(X_train, axis=2), y=y_train, batch_size=64, epochs=10, verbose=1,
                  validation_split=0.1,
                  shuffle=True)
        y_pred = model.predict(np.expand_dims(X_test, axis=2))
        qmse = metrics.mean_squared_error(y_test, y_pred)
        qrmse = np.sqrt(qmse)
        print('Qmse = ', qmse, " QRmse=", qrmse)
        qcorr = qmse / qrmse
        rsltdict.update({'qpfmse': qmse, 'qpfrmse': qrmse, 'qpfcorr': qcorr})
        # accuracy = metrics.accuracy_score(y_test.round(), y_pred.round(),normalize=False)
        # print('QPF accuracy ',accuracy)
        plt.close()'''
        return rsltdict

