import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Assuming 'Effective Date' is datetime format, if not, convert it to datetime
df['Effective Date'] = pd.to_datetime(df['Effective Date'])

# Convert 'Exchange Rate' to numeric (handling commas and errors)
df['Exchange Rate'] = pd.to_numeric(df['Exchange Rate'].replace(',', ''), errors='coerce')

# Drop rows with non-numeric values in 'Exchange Rate'
df = df.dropna(subset=['Exchange Rate'])

# Create a unique set of case-insensitive currency names, use camel case
unique_currency_set = set()
for currency in df['Country - Currency Description']:
    unique_currency_set.add(currency.title().replace(" ", ""))

# Convert the set back to a list
currencies = list(unique_currency_set)

# Function to predict exchange rate for a given currency
def predict_exchange_rate(currency):
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
    print(f'RMSE for {currency}: {rmse}')

    # Plotting with adjusted figure size
    fig, ax = plt.subplots(figsize=(6, 4))  # Adjust the figure size as needed
    ax.plot(train.index, train['Exchange Rate'], label='Training Data')
    ax.plot(test.index, test['Exchange Rate'], label='Actual Exchange Rate')
    ax.plot(test.index, predictions, label='Predicted Exchange Rate')
    ax.set_title(f'Exchange Rate Prediction for {currency}')
    ax.legend()

    return fig, rmse

# Function to handle suggestion selection and completion
def select_suggestion(event):
    selected_index = suggestion_listbox.nearest(event.y)
    if selected_index >= 0:
        selected_suggestion = suggestion_listbox.get(selected_index)
        completed_text = selected_suggestion
        search_var.set(completed_text)
        search_entry.icursor(len(completed_text))  # Move cursor to end of completed text
        search_entry.focus_set()  # Set focus back to entry box
        update_plot(completed_text)
        suggestion_listbox.place_forget()  # Hide the suggestion list

def update_plot(text):
    if text in currencies:
        plt.clf()  # Clear the previous plot
        fig, rmse = predict_exchange_rate(text)
        canvas.figure = fig
        canvas.draw()
        rmse_label['text'] = f'RMSE: {rmse}'

# Function to update plot based on entry box content
def update_plot_from_entry(event):
    entered_text = search_var.get()
    # Update the plot if the entered text is a valid currency
    update_plot(entered_text)

    suggestions = [currency for currency in currencies if entered_text in currency.lower()]

    # Clear previous suggestions
    suggestion_listbox.delete(0, tk.END)

    # Add new suggestions
    for suggestion in suggestions:
        suggestion_listbox.insert(tk.END, suggestion)

    # Show the Listbox if there are suggestions, hide it otherwise
    if suggestions:
        suggestion_listbox.place(x=search_entry.winfo_x(), y=search_entry.winfo_y() + search_entry.winfo_height())
    else:
        suggestion_listbox.place_forget()
    

# Create the Tkinter root window
root = tk.Tk()
root.title("Exchange Rate Prediction")

# Add an entry field for search
search_label = tk.Label(root, text="Enter Currency:")
search_label.pack(side=tk.LEFT)
search_var = tk.StringVar()
search_entry = tk.Entry(root, textvariable=search_var)
search_entry.pack(side=tk.LEFT)
search_entry.bind('<KeyRelease>', update_plot_from_entry)

# Add RMSE label
rmse_label = tk.Label(root, text="")
rmse_label.pack(side=tk.BOTTOM)

# Create a Listbox for suggestions
suggestion_listbox = tk.Listbox(root, selectmode=tk.SINGLE, exportselection=0)
suggestion_listbox.place_forget()  # Initially hide the Listbox
suggestion_listbox.bind('<ButtonRelease-1>', select_suggestion)

# Embed the Matplotlib figure in the Tkinter window
canvas = FigureCanvasTkAgg(plt.figure(), master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Set focus to the Entry widget when the window gains focus
root.bind("<FocusIn>", lambda event: search_entry.focus_set())

# Start the Tkinter event loop
root.mainloop()
