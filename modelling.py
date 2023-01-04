import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


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


run_model_one()
