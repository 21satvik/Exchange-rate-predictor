from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def predict_exchange_rate_using_GBM(currency, df):
    currency_df = df[df['Country - Currency Description'].str.lower() == currency.lower()]

    # Sort the data by 'Effective Date'
    currency_df = currency_df.sort_values(by='Effective Date')

    # Set 'Effective Date' as the index
    currency_df.set_index('Effective Date', inplace=True)

    # Feature engineering: You may need to add more features based on your dataset
    currency_df['year'] = currency_df.index.year
    currency_df['month'] = currency_df.index.month
    currency_df['day'] = currency_df.index.day

    # Split the data into training and testing sets
    train_size = int(len(currency_df) * 0.8)
    train, test = currency_df[:train_size], currency_df[train_size:]

    # Define features and target variable
    features = ['year', 'month', 'day']  # Add more features as needed
    target = 'Exchange Rate'

    # Train Gradient Boosting Regressor
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)  # Adjust parameters as needed
    model.fit(train[features], train[target])

    # Make predictions
    predictions = model.predict(test[features])

    # Calculate RMSE (Root Mean Squared Error) for evaluation
    rmse = mean_squared_error(test[target], predictions, squared=False)
    print(f'RMSE for {currency} using Gradient Boosting Machine: {rmse}')

    # Plotting with adjusted figure size
    fig, ax = plt.subplots(figsize=(6, 4))  # Adjust the figure size as needed
    ax.plot(train.index, train[target], label='Training Data')
    ax.plot(test.index, test[target], label='Actual Exchange Rate')
    ax.plot(test.index, predictions, label='Predicted Exchange Rate')
    ax.set_title(f'Exchange Rate Prediction for {currency}')
    ax.legend()

    return fig, rmse
