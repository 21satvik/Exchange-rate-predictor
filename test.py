import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from ARIMA_model import predict_exchange_rate_using_ARIMA
from Integrated_model import predict_exchange_rate_using_integrated_model
from GBM_model import predict_exchange_rate_using_GBM
from randomforest_model import predict_exchange_rate_using_random_forest

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
    unique_currency_set.add(currency.title())

# Convert the set back to a list
currencies = list(unique_currency_set)

# Available models
available_models = ['ARIMA', 'Random Forest','GBM', 'Integrated']

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

# Function to update the plot
def update_plot(text):
    if text in currencies:
        plt.clf()  # Clear the previous plot
        selected_model = model_var.get()
        if selected_model == 'ARIMA':
            fig, rmse = predict_exchange_rate_using_ARIMA(text, df)
        elif selected_model == 'Random Forest':
            fig, rmse = predict_exchange_rate_using_random_forest(text, df)
        elif selected_model == 'GBM':
            fig, rmse = predict_exchange_rate_using_GBM(text, df)
        elif selected_model == 'Integrated':
            fig, rmse = predict_exchange_rate_using_integrated_model(text, df)
        else:
            return

        canvas.figure = fig
        canvas.draw()
        rmse_label['text'] = f'RMSE: {rmse}'

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

# Add a dropdown menu for model selection
search_label = tk.Label(root, text="Select Model")
search_label.pack(side=tk.TOP)
model_var = tk.StringVar()
model_var.set(available_models[0])  # Set the default model
model_dropdown = ttk.Combobox(root, textvariable=model_var, values=available_models)
model_dropdown.pack(side=tk.TOP)
model_dropdown.bind("<<ComboboxSelected>>", lambda event: update_plot(search_var.get()))

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
