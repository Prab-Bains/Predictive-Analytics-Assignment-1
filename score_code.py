import pandas as pd
from sklearn.impute import KNNImputer

# Replace path where the input dataset is
PATH = "/Users/prabhjeetbains/Desktop/BCIT/Term 3/Predictive Modelling/Data sets/"
CSV_DATA = "customer_data_mystery.csv"

# Reads the dataset and saves it in a variable
dataset = pd.read_csv(PATH + CSV_DATA, skiprows=1, encoding="ISO-8859-1", sep=',',
                      names=('Year_Birth', 'Education', 'Marital_Status', 'Kidhome', 'Teenhome',
                             'Dt_Customer', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                             'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                             'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3',
                             'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain',
                             'Z_CostContact', 'Z_Revenue', 'Response'))

# Display all columns of the data frame.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

X = dataset[['Teenhome', 'MntWines', 'MntMeatProducts',
             'MntSweetProducts', 'NumWebPurchases',
             'NumStorePurchases', 'NumWebVisitsMonth',
             'Z_CostContact', 'Z_Revenue']]

imputer = KNNImputer(n_neighbors=5)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

df = pd.DataFrame(X, columns=['Teenhome', 'MntWines', 'MntMeatProducts',
                              'MntSweetProducts', 'NumWebPurchases',
                              'NumStorePurchases', 'NumWebVisitsMonth',
                              'Z_CostContact', 'Z_Revenue'])


# uses the coefficients from the model summary to predict the customers income
def predict_customer_income(input_df):
    customer_income_predictions_list = []

    for index in range(0, len(input_df)):
        predicted_income = 4394.9539 * input_df.iloc[index]['Teenhome'] + \
                           18.7240 * input_df.iloc[index]['MntWines'] + \
                           24.1679 * input_df.iloc[index]['MntMeatProducts'] + \
                           52.0634 * input_df.iloc[index]['MntSweetProducts'] + \
                           821.8285 * input_df.iloc[index]['NumWebPurchases'] + \
                           465.9154 * input_df.iloc[index]['NumStorePurchases'] - \
                           3054.4522 * input_df.iloc[index]['NumWebVisitsMonth'] + \
                           1125.4987 * input_df.iloc[index]['Z_CostContact'] + \
                           4126.8287 * input_df.iloc[index]['Z_Revenue']

        customer_income_predictions_list.append(predicted_income)

    return customer_income_predictions_list


predictions = predict_customer_income(df)
dfPredictions = pd.DataFrame()
dfPredictions['Predictions'] = predictions

# Writes all the predictions to the given file
dfPredictions.to_csv('customer_data_predictions.csv', index=False)
