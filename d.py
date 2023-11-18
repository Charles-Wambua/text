from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

dataset = load_dataset("cnn_dailymail", "2.0.0")



train_data = dataset["train"]
validation_data = dataset["validation"]


train_articles = [example["article"] for example in train_data]
train_summaries = [example["highlights"] for example in train_data]

validation_articles = [example["article"] for example in validation_data]
validation_summaries = [example["highlights"] for example in validation_data]


tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
train_encodings = tokenizer(train_articles, train_summaries, truncation=True, padding=True, return_tensors="pt")
validation_encodings = tokenizer(validation_articles, validation_summaries, truncation=True, padding=True, return_tensors="pt")

class SummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = SummarizationDataset(train_encodings)
validation_dataset = SummarizationDataset(validation_encodings)


model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=2, shuffle=False)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    average_train_loss = train_loss / len(train_loader)

    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(validation_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            validation_loss += loss.item()

    average_validation_loss = validation_loss / len(validation_loader)

    print(f"Epoch {epoch + 1}/{num_epochs} - Average Training Loss: {average_train_loss:.4f}, Average Validation Loss: {average_validation_loss:.4f}")


model.save_pretrained("fine_tuned_model")