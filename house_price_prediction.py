import numpy as np
import pandas as pd
from typing import NoReturn
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
from ex1_208936625.linear_regression import LinearRegression
import plotly.io as pio

pio.templates.default = "simple_white"

global_x_train = None


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    X = X.dropna().drop_duplicates()
    X = X.drop(columns=["id", "date", "sqft_living15", "sqft_lot15"])
    X["price"] = y
    positive_features = ["price", "sqft_living", "sqft_lot", "sqft_above", "yr_built", "floors", "bathrooms"]
    for feature in positive_features:
        X = X[X[feature] > 0]
    X = X[X["waterfront"].isin([0, 1]) & X["view"].isin(range(5)) & (X["yr_renovated"] >= 0) & (X["sqft_basement"] >= 0)
          & X["condition"].isin(range(1, 6)) & X["grade"].isin(range(1, 14)) & X["yr_built"] < X["yr_renovated"]]
    __grades_design(X)
    X["renovated_lately"] = np.where(X["yr_renovated"] <= np.percentile(X.yr_renovated.unique(), 80), 1, 0)
    y_res = X["price"]
    X = X.drop(columns=["price", "yr_renovated"])
    return X, y_res


# design the grades to be with more impact
def __grades_design(data):
    data.loc[(data['grade'] >= 1) & (data['grade'] <= 3), 'grade'] = 1
    data.loc[(data['grade'] >= 4) & (data['grade'] <= 7), 'grade'] = 2
    data.loc[(data['grade'] >= 8) & (data['grade'] <= 10), 'grade'] = 3
    data.loc[(data['grade'] >= 11) & (data['grade'] <= 13), 'grade'] = 4


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    __grades_design(X)
    X["renovated_lately"] = np.where(X["yr_renovated"] <= np.percentile(X.yr_renovated.unique(), 80), 1, 0)
    X = X.drop(columns=["id", "date", "yr_renovated"])
    X = X.reindex(columns=global_x_train.columns, fill_value=0)

    # Replace NA values with the mean value of the feature
    for col in global_x_train.columns:
        avg = global_x_train[col].mean()
        X[col] = X[col].apply(lambda x: avg if x < 0 or np.isnan(x) else x)
    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    y_std = y.std()
    for feature in X:
        pearson_corr = X[feature].cov(y) / (X[feature].std() * y_std)
        fig = px.scatter(pd.DataFrame({'x': X[feature], 'y': y}), x="x", y="y",
                         title=f"Pearson Correlation: {pearson_corr}",
                         labels={"x": f"{feature}", "y": "Price"})
        fig.write_image(output_path + "/Q3.1.4_feature_evaluation_" + str(feature) + ".png")


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=0)

    # Question 3 - preprocessing of housing prices train dataset
    X_train, y_train = preprocess_train(X_train, y_train)
    global_x_train = X_train

    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_train, y_train)

    # Question 5 - preprocess the test data
    X_test = preprocess_test(X_test)

    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    results = {'percentages': [], 'averages': [], 'y_lower': [], 'y_upper': []}
    LR = LinearRegression(include_intercept=True)
    for perc in range(10, 101):
        results['percentages'].append(perc)
        curr_losses = []
        for times in range(10):
            X_sample = X_train.sample(frac=perc / 100)
            y_sample = y_train.loc[X_sample.index]
            LR.fit(X_sample.to_numpy(), y_sample.to_numpy())
            curr_losses.append(LR.loss(X_test.to_numpy(), y_test.to_numpy()))
        cur_mean = np.mean(curr_losses)
        cur_std = np.std(curr_losses)
        results['averages'].append(cur_mean)
        results['y_lower'].append(cur_mean - (2 * cur_std))
        results['y_upper'].append(cur_mean + (2 * cur_std))
        x_axis = results['percentages']
        y_axis = results['averages']
    fig = go.Figure([go.Scatter(x=x_axis, y=y_axis, line=dict(color='rgb(0,100,80)'), name='Mean loss for percent',
                                mode="markers+lines"),
                     go.Scatter(x=x_axis, y=y_axis, line=dict(color='rgb(0,100,80)'), showlegend=False, mode='lines'),
                     go.Scatter(x=x_axis, y=y_axis, line=dict(color='rgb(0,100,80)'), showlegend=False, mode='lines',
                                fill='tonexty')],
                    layout=go.Layout(title="Loss of model by percent of train data",
                                     xaxis={"title": "percent of train data"}, yaxis={"title": "Loss of model"}))
    fig.write_image("Q3.1.6_Fit_Model_Over_Percentages.png")