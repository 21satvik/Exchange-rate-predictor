# Currency Exchange Rate Predictor

This project focuses on predicting currency exchange rates using machine learning models, specifically a Random Forest Regressor and an ARIMA model. The project also includes a user interface for data visualization implemented with Tkinter.

## Overview

- **Objective:** Predict exchange rates and provide a user-friendly interface for visualization.
- **Models Used:**
  - Random Forest Regressor
  - ARIMA (Autoregressive Integrated Moving Average)
- **Visualization Tool:** Tkinter GUI with Matplotlib integration.

## Project Structure

- **`data/`:** Directory for storing dataset files.
- **`models/`:** Contains scripts for training and using machine learning models.
- **`visualization/`:** Tkinter GUI implementation and visualization scripts.
- **`utils/`:** Utility scripts for data processing, file handling, and preprocessing.
- **`main.py`:** Main script to run the Tkinter GUI and execute the currency prediction.

## Dependencies

- Python 3.x
- Libraries:
  - scikit-learn
  - statsmodels
  - matplotlib
  - pandas
  - tkinter

## Setup

1. Install dependencies using: `pip install -r requirements.txt`
2. Run the main script: `python main.py`

## Usage

1. Launch the Tkinter GUI by running `main.py`.
2. Enter the currency for which you want to predict exchange rates.
3. View the predicted exchange rates along with a comparison to the actual rates in the plotted graph.

## Contributing

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Create a pull request.
