import tkinter as tk
from tkinter import ttk, scrolledtext
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pyperclip
from threading import Thread

# Load tokenizer and other model-related data (adjust based on your model)
def load_tokenizer_and_model():
    tokenizer = Tokenizer()
    # Load your tokenizer using the correct path
    tokenizer_json_path = 'path_to_your_tokenizer.json'
    with open(tokenizer_json_path, 'r') as f:
        data = f.read()
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

    # Get the maximum length of input sequences (adjust based on your model)
    max_len_article = 100  # Placeholder value, replace with the actual value

    # Placeholder: Dummy model for illustration (replace with loading your actual model)
    # Load your model using the correct path
    model = load_model('path_to_your_model.h5')

    return tokenizer, model, max_len_article

# Function to summarize text using a pre-trained model
def summarize_text():
    input_text = input_textbox.get("1.0", tk.END).strip()

    # Preprocess the input text
    tokenized_input = tokenizer.texts_to_sequences([input_text])
    padded_input = pad_sequences(tokenized_input, maxlen=max_len_article, padding='post')

    # Get the summarized text from the model
    summarized_sequence = model.predict(padded_input)[0]

    # Decode the output sequence
    decoded_summary = ' '.join([index_word.get(token, '') for token in summarized_sequence])

    # Update the output textbox with the summarized text
    output_textbox.delete("1.0", tk.END)
    output_textbox.insert(tk.END, decoded_summary)

    # Enable the copy button
    copy_button['state'] = tk.NORMAL
    # Stop the loading animation
    loading_label.config(text="")

# Function to copy the summarized text to the clipboard
def copy_to_clipboard():
    summarized_text = output_textbox.get("1.0", tk.END).strip()
    pyperclip.copy(summarized_text)

# Function to show loading animation while summarizing
def show_loading_animation():
    loading_label.config(text="Summarizing...", font=('Helvetica', 10, 'italic'))

# Create the main Tkinter window
window = tk.Tk()
window.title("Text Summarizer")

# Load tokenizer and model
tokenizer, model, max_len_article = load_tokenizer_and_model()

# Create and configure widgets
input_label = ttk.Label(window, text="Paste Your Text Below:")
input_textbox = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=40, height=10, font=('Helvetica', 12))
summarize_button = ttk.Button(window, text="Summarize", command=lambda: Thread(target=summarize_text).start())
loading_label = ttk.Label(window, text="", font=('Helvetica', 10, 'italic'))
output_label = ttk.Label(window, text="Summarized Text:")
output_textbox = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=40, height=10, font=('Helvetica', 12), state=tk.DISABLED)
copy_button = ttk.Button(window, text="Copy to Clipboard", command=copy_to_clipboard, state=tk.DISABLED)

# Place widgets on the grid
input_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
input_textbox.grid(row=1, column=0, padx=10, pady=5, columnspan=2)
summarize_button.grid(row=2, column=0, pady=5, sticky='w')
loading_label.grid(row=2, column=1, pady=5, sticky='w')
output_label.grid(row=3, column=0, padx=10, pady=5, sticky='w')
output_textbox.grid(row=4, column=0, padx=10, pady=5, columnspan=2)
copy_button.grid(row=5, column=0, pady=5, sticky='w')

# Set grid weights to allow resizing
window.grid_rowconfigure(4, weight=1)
window.grid_columnconfigure(1, weight=1)

# Start the Tkinter event loop
window.mainloop()
