
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from ex1_208936625.polynomial_fitting import PolynomialFitting


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Convert the 'Date' column to datetime, coercing invalid dates to NaT
    data = data.dropna(subset=['Date'])     # Remove rows with invalid dates
    data = data[(data["Temp"] > -50) & (data["Temp"] < 50)]
    data['DayOfYear'] = data['Date'].dt.dayofyear
    return data


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country
    israel_data = df[df['Country'] == 'Israel']
    israel_data['Year'] = israel_data['Year'].astype(str)
    scatter_plot = px.scatter(israel_data, x="DayOfYear", y="Temp", color="Year")
    scatter_plot.write_image("Q3.2.3_city_temp.png")
    bar_plot = px.bar(israel_data.groupby(['Month'], as_index=False).agg(std=('Temp', 'std')),
                      y='std', x=list(range(1, 13))).update_layout(xaxis_title='Month')
    bar_plot.write_image("Q3.2.3_temperature_deviation.png")

    # Question 4 - Exploring differences between countries
    line_plot = px.line(df.groupby(['Country', 'Month'], as_index=False).agg(mean=('Temp', 'mean'),
                     std=('Temp', 'std')), x='Month', y='mean', error_y='std', color='Country')
    line_plot.update_layout(yaxis_title='Mean Temperature')
    line_plot.write_image("Q3.2.4_avg_temp_in_diff_countries.png")
    
    # Question 5 - Fitting model for different values of `k`
    X_train, X_test, y_train, y_test = train_test_split(israel_data['DayOfYear'], israel_data['Temp'], test_size=0.75)
    losses = []
    for k in range(1, 11):
        poly_fit = PolynomialFitting(k)
        poly_fit.fit(X_train.to_numpy(), y_train.to_numpy())
        losses.append(np.round(poly_fit.loss(X_test.to_numpy(), y_test.to_numpy()), 2))
        print('k = ' + str(k) + ': ' + str(losses[k - 1]))
    bar_plot_q5 = px.bar(x=range(1, 11), y=losses, text=losses).update_layout(xaxis_title="Degree",
                                                                              yaxis_title="Loss Error")
    bar_plot_q5.write_image("Q3.2.5_loss_as_function_ofDegree.png")

    # Question 6 - Evaluating fitted model on different countries
    countries = {'Country': [], 'Loss': []}
    israel_model = PolynomialFitting(5)   # Best k found
    israel_model.fit(israel_data['DayOfYear'].to_numpy(), israel_data['Temp'].to_numpy())
    for country in df['Country'].unique():
        if country == "Israel":
            continue
        countries['Country'].append(country)
        df_country = df[df['Country'] == country]
        country_loss = israel_model.loss(df_country['DayOfYear'].to_numpy(), df_country['Temp'].to_numpy())
        countries['Loss'].append(country_loss)
    bar_plot_q6 = px.bar(pd.DataFrame(countries), x='Country', y='Loss')
    bar_plot_q6.write_image("EX1_Q3.2.6_loss_over_countries_model_only_israel.png")
