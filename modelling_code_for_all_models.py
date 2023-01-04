import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

LOWER_PERCENTILE = 0.00025
UPPER_PERCENTILE = 0.99925


def plot_prediction_vs_actual(title, y_test_variable, predictions_variable):
    plt.scatter(y_test_variable, predictions_variable)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title('Predicted (Y) vs. Actual (X): ' + title)
    plt.plot([y_test_variable.min(), y_test_variable.max()], [y_test_variable.min(), y_test_variable.max()], 'k--')
    plt.figure(figsize=(20, 20))
    plt.show()


def plotResidualsVsActual(title, y_test, predictions):
    dfPrecisions = pd.DataFrame(data={"predictions": predictions})
    residuals = dfPrecisions['predictions'] - y_test['Income']
    plt.scatter(y_test, residuals, label='Residuals vs Actual')
    plt.xlabel("Actual")
    plt.ylabel("Residual")
    plt.title('Error Residuals (Y) vs. Actual (X): ' + title)
    plt.plot([y_test.min(), y_test.max()], [0, 0], 'k--')
    plt.show()


PATH = "/Users/prabhjeetbains/Desktop/BCIT/Term 3/Predictive Modelling/Data sets/"
CSV_DATA = "customer_data.csv"

dataset = pd.read_csv(PATH + CSV_DATA, skiprows=1, encoding="ISO-8859-1", sep=',',
                      names=('Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome',
                             'Dt_Customer', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                             'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                             'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3',
                             'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain',
                             'Z_CostContact', 'Z_Revenue', 'Response'))

# Display all columns of the data frame.

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def run_model_one():
    print('----------------- First Model, Only Numeric Data -----------------\n')

    # Original data with numeric columns.
    X = dataset[['Teenhome', 'MntWines', 'MntMeatProducts',
                 'MntSweetProducts', 'NumWebPurchases',
                 'NumStorePurchases', 'NumWebVisitsMonth',
                 'Z_CostContact', 'Z_Revenue']]

    y = dataset[['Income']]

    imputer = KNNImputer(n_neighbors=5)
    y = pd.DataFrame(imputer.fit_transform(y), columns=y.columns)
    X = sm.add_constant(X)

    kfold = KFold(n_splits=6, shuffle=True)
    rmses = []

    for train_index, test_index in kfold.split(X):
        # use index lists to isolate rows for train and test sets.
        X_train = X.loc[X.index.intersection(train_index), :]
        X_test = X.loc[X.index.intersection(test_index), :]
        y_train = y.loc[y.index.intersection(train_index), :]
        y_test = y.loc[y.index.intersection(test_index), :]

        model = sm.OLS(y_train, X_train).fit()
        predictions = model.predict(X_test)  # make the predictions by the model

        mse = mean_squared_error(predictions, y_test)
        rmse = np.sqrt(mse)
        rmses.append(rmse)

    avgRMSE = np.mean(rmses)

    print(model.summary())
    print("Average Root Mean Squared Error Across All Folds: " + str(avgRMSE))
    # plotResidualsVsActual("Customer Incomes", y_test, predictions)


########################################################################################################################
########################################################################################################################


def remove_outliers_and_impute_missing_data(df, col_name):
    # Find the inner quartiles of each column
    upper = df[col_name].quantile(UPPER_PERCENTILE)
    lower = df[col_name].quantile(LOWER_PERCENTILE)

    # Removes outliers from data frame
    df = df[(df[col_name] >= lower) & (df[col_name] <= upper)]
    return df


def run_model_two():
    print('\n----------------- Second Model, Removed Outliers and Insignificant Variables -----------------\n')

    # Remove Outliers
    a = remove_outliers_and_impute_missing_data(dataset[['Year_Birth']], 'Year_Birth')
    b = remove_outliers_and_impute_missing_data(dataset[['Kidhome']], 'Kidhome')
    c = remove_outliers_and_impute_missing_data(dataset[['Teenhome']], 'Teenhome')
    d = remove_outliers_and_impute_missing_data(dataset[['Recency']], 'Recency')
    e = remove_outliers_and_impute_missing_data(dataset[['MntWines']], 'MntWines')
    f = remove_outliers_and_impute_missing_data(dataset[['MntFruits']], 'MntFruits')
    g = remove_outliers_and_impute_missing_data(dataset[['MntMeatProducts']], 'MntMeatProducts')
    h = remove_outliers_and_impute_missing_data(dataset[['MntFishProducts']], 'MntFishProducts')
    i = remove_outliers_and_impute_missing_data(dataset[['MntSweetProducts']], 'MntSweetProducts')
    j = remove_outliers_and_impute_missing_data(dataset[['MntGoldProds']], 'MntGoldProds')
    k = remove_outliers_and_impute_missing_data(dataset[['NumDealsPurchases']], 'NumDealsPurchases')
    l = remove_outliers_and_impute_missing_data(dataset[['NumWebPurchases']], 'NumWebPurchases')
    m = remove_outliers_and_impute_missing_data(dataset[['NumCatalogPurchases']], 'NumCatalogPurchases')
    n = remove_outliers_and_impute_missing_data(dataset[['NumStorePurchases']], 'NumStorePurchases')
    o = remove_outliers_and_impute_missing_data(dataset[['NumWebVisitsMonth']], 'NumWebVisitsMonth')
    p = remove_outliers_and_impute_missing_data(dataset[['AcceptedCmp3']], 'AcceptedCmp3')
    q = remove_outliers_and_impute_missing_data(dataset[['AcceptedCmp4']], 'AcceptedCmp4')
    r = remove_outliers_and_impute_missing_data(dataset[['AcceptedCmp5']], 'AcceptedCmp5')
    s = remove_outliers_and_impute_missing_data(dataset[['AcceptedCmp1']], 'AcceptedCmp1')
    t = remove_outliers_and_impute_missing_data(dataset[['AcceptedCmp2']], 'AcceptedCmp2')
    u = remove_outliers_and_impute_missing_data(dataset[['Complain']], 'Complain')
    v = remove_outliers_and_impute_missing_data(dataset[['Z_CostContact']], 'Z_CostContact')
    w = remove_outliers_and_impute_missing_data(dataset[['Z_Revenue']], 'Z_Revenue')
    x = remove_outliers_and_impute_missing_data(dataset[['Response']], 'Response')

    dataset_filtered = pd.concat(
        [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, dataset[['Income']]], axis=1)

    X = dataset_filtered[['Teenhome', 'MntWines', 'MntMeatProducts',
                          'MntSweetProducts', 'NumDealsPurchases', 'NumWebPurchases',
                          'NumWebVisitsMonth',
                          'Z_CostContact', 'Z_Revenue']]

    y = dataset_filtered[['Income']]

    # Impute the data
    imputer = KNNImputer(n_neighbors=6)
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    y = pd.DataFrame(imputer.fit_transform(y), columns=y.columns)

    X = sm.add_constant(X)

    kfold = KFold(n_splits=10, shuffle=True)

    rmses = []
    for train_index, test_index in kfold.split(X):
        # use index lists to isolate rows for train and test sets.
        X_train = X.loc[X.index.intersection(train_index), :]
        X_test = X.loc[X.index.intersection(test_index), :]
        y_train = y.loc[y.index.intersection(train_index), :]
        y_test = y.loc[y.index.intersection(test_index), :]

        model = sm.OLS(y_train, X_train).fit()
        predictions = model.predict(X_test)  # make the predictions by the model

        mse = mean_squared_error(predictions, y_test)
        rmse = np.sqrt(mse)
        rmses.append(rmse)

    avgRMSE = np.mean(rmses)
    print(model.summary())
    print("Average Root Mean Squared Error Across All Folds: " + str(avgRMSE))


########################################################################################################################
########################################################################################################################

def run_model_three():
    print('\n----------------- Third Model, Binned and Dummy Variables -----------------\n')

    tempDf1 = dataset[['Education']]  # need to get dummy variables for

    tempDf2 = dataset[['Year_Birth', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
                       'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3',
                       'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain',
                       'Z_CostContact', 'Z_Revenue', 'Response']]  # don't need dummies

    tempDf3 = dataset[['Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
                       'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3',
                       'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain',
                       'Z_CostContact', 'Z_Revenue',
                       'Response']]  # Remove 'Year_birth' column from tempDf2 for further binning columns to join.

    dummyDf = pd.get_dummies(tempDf1, columns=['Education'])  # get dummies

    tempDf2['Year_Birth'] = pd.cut(x=tempDf2['Year_Birth'],
                                   bins=[0, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000])
    tempDf = tempDf2[['Year_Birth']]
    dummyDf2 = pd.get_dummies(tempDf, columns=['Year_Birth'])

    X_before_remove_insignificant_ones = pd.concat(([tempDf3, dummyDf, dummyDf2]),
                                                   axis=1)

    X = X_before_remove_insignificant_ones[
        ['Teenhome', 'MntWines', 'MntMeatProducts',
         'MntSweetProducts', 'NumWebPurchases',
         'NumWebVisitsMonth',
         'Z_CostContact', 'Z_Revenue', 'Education_Basic', 'Year_Birth_(1980, 1990]']]

    y = dataset[['Income']]

    # Impute values for non-value cell.
    imputer = KNNImputer(n_neighbors=5)
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    y = pd.DataFrame(imputer.fit_transform(y), columns=y.columns)

    X = sm.add_constant(X)

    kfold = KFold(n_splits=6, shuffle=True)

    rmses = []
    for train_index, test_index in kfold.split(X):
        # use index lists to isolate rows for train and test sets.
        X_train = X.loc[X.index.intersection(train_index), :]
        X_test = X.loc[X.index.intersection(test_index), :]
        y_train = y.loc[y.index.intersection(train_index), :]
        y_test = y.loc[y.index.intersection(test_index), :]

        model = sm.OLS(y_train, X_train).fit()
        predictions = model.predict(X_test)  # make the predictions by the model

        mse = mean_squared_error(predictions, y_test)
        rmse = np.sqrt(mse)
        rmses.append(rmse)

    avgRMSE = np.mean(rmses)
    print(model.summary())
    print("Average Root Mean Squared Error Across All Folds: " + str(avgRMSE))
    # Display all columns of the data frame.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)


run_model_one()
run_model_two()
run_model_three()
