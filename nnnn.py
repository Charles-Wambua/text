import tkinter as tk
from tkinter import scrolledtext
from transformers import pipeline

class TextSummarizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Text Summarizer App")

        self.create_widgets()

    def create_widgets(self):
        
        self.input_text = scrolledtext.ScrolledText(self.master, width=60, height=10, wrap=tk.WORD)
        self.input_text.pack(pady=10)

        self.summarize_button = tk.Button(self.master, text="Summarize", command=self.summarize_text)
        self.summarize_button.pack(pady=5)

        self.output_text = scrolledtext.ScrolledText(self.master, width=60, height=10, wrap=tk.WORD)
        self.output_text.pack(pady=10)

       
        self.copy_button = tk.Button(self.master, text="Copy", command=self.copy_text)
        self.copy_button.pack(side=tk.LEFT, padx=5)

     
        self.save_button = tk.Button(self.master, text="Save", command=self.save_text)
        self.save_button.pack(side=tk.LEFT, padx=5)

    def summarize_text(self):
        input_text = self.input_text.get("1.0", tk.END)
        summarization_pipeline = pipeline("summarization")
        summary = summarization_pipeline(input_text)

   
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, summary[0]['summary_text'])

    def copy_text(self):
        summary_text = self.output_text.get("1.0", tk.END)
        self.master.clipboard_clear()
        self.master.clipboard_append(summary_text)
        self.master.update()

    def save_text(self):
        summary_text = self.output_text.get("1.0", tk.END)
        with open("summarized_text.txt", "w") as file:
            file.write(summary_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = TextSummarizerApp(root)
    root.mainloop()
