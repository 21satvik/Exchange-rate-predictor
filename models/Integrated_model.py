from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from ARIMA_model import predict_exchange_rate_using_ARIMA
from GBM_model import predict_exchange_rate_using_GBM
from randomforest_model import predict_exchange_rate_using_random_forest

def predict_exchange_rate_using_integrated_model(currency, df):
    arima_fig, arima_rmse = predict_exchange_rate_using_ARIMA(currency, df)
    randomforest_fig, randomforest_rmse = predict_exchange_rate_using_random_forest(currency, df)
    gbm_fig, gbm_rmse = predict_exchange_rate_using_GBM(currency, df)

    # Get ARIMA and Random Forest predictions
    arima_predictions = arima_fig.gca().lines[-1].get_ydata()  # Extract predictions from ARIMA figure
    randomforest_predictions = randomforest_fig.gca().lines[-1].get_ydata()  # Extract predictions from Random Forest figure
    gbm_predictions = gbm_fig.gca().lines[-1].get_ydata()  # Extract predictions from GBM figure

    # Calculate integrated predictions by averaging
    integrated_predictions = (arima_predictions + randomforest_predictions + gbm_predictions) / 3

    # Plotting integrated predictions
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(arima_fig.gca().lines[0].get_xdata(), arima_fig.gca().lines[0].get_ydata(), label='Training Data (ARIMA)')
    ax.plot(arima_fig.gca().lines[1].get_xdata(), arima_fig.gca().lines[1].get_ydata(), label='Actual Exchange Rate (ARIMA)')
    ax.plot(arima_fig.gca().lines[2].get_xdata(), integrated_predictions, label='Integrated Predicted Exchange Rate')
    ax.set_title(f'Exchange Rate Prediction for {currency}')
    ax.legend()

    # Calculate integrated RMSE
    integrated_rmse = mean_squared_error(arima_fig.gca().lines[1].get_ydata(), integrated_predictions, squared=False)
    print(f'RMSE for {currency} using Integrated Model: {integrated_rmse}')

    return fig, integrated_rmse
