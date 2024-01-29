from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def predict_exchange_rate_using_ARIMA(currency, df):
    currency_df = df[df['Country - Currency Description'].str.lower() == currency.lower()]

    # Sort the data by 'Effective Date'
    currency_df = currency_df.sort_values(by='Effective Date')

    # Set 'Effective Date' as the index
    currency_df.set_index('Effective Date', inplace=True)

    # Split the data into training and testing sets
    train_size = int(len(currency_df) * 0.8)
    train, test = currency_df[:train_size], currency_df[train_size:]

    # Convert the 'Exchange Rate' column to a one-dimensional array
    train_exchange_rate = train['Exchange Rate'].values

    # Fit ARIMA model
    model = ARIMA(train_exchange_rate, order=(5, 1, 0))  # Adjust order as needed
    model_fit = model.fit()

    # Convert the 'Exchange Rate' column to a one-dimensional array for testing
    test_exchange_rate = test['Exchange Rate'].values

    # Make predictions
    predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

    # Calculate RMSE (Root Mean Squared Error) for evaluation
    rmse = mean_squared_error(test_exchange_rate, predictions, squared=False)
    print(f'RMSE for {currency} using ARIMA: {rmse}')

    # Plotting with adjusted figure size
    fig, ax = plt.subplots(figsize=(6, 4))  # Adjust the figure size as needed
    ax.plot(train.index, train['Exchange Rate'], label='Training Data')
    ax.plot(test.index, test['Exchange Rate'], label='Actual Exchange Rate')
    ax.plot(test.index, predictions, label='Predicted Exchange Rate')
    ax.set_title(f'Exchange Rate Prediction for {currency}')
    ax.legend()

    return fig, rmse