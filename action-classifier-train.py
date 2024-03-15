import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split

df = pd.read_csv("~/action-classifier/action-classifier-dataset.csv")

test_split = 0.1
train_df, test_df = train_test_split(
    df,
    test_size=test_split,
)
print(f"Number of rows in training set: {len(train_df)}")
print(f"Number of rows in test set: {len(test_df)}")

# Define a list of columns that should not be chosen as label columns
not_chosen_columns = ['statement']

# Select label columns that are not in the list of not chosen columns
label_columns = [col for col in df.columns if col not in not_chosen_columns]

# Create a new DataFrame containing only the selected label columns
df_labels_train = train_df[label_columns]
df_labels_test = test_df[label_columns]


# Convert the label columns to lists for each row
labels_list_train = df_labels_train.values.tolist()
labels_list_test = df_labels_test.values.tolist()

train_texts = train_df['statement'].tolist()
train_labels = labels_list_train

eval_texts = test_df['statement'].tolist()
eval_labels = labels_list_test

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

train_encodings = tokenizer(train_texts, padding="max_length", truncation=True, max_length=512)
eval_encodings = tokenizer(eval_texts, padding="max_length", truncation=True, max_length=512)


class TextClassifierDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

train_dataset = TextClassifierDataset(train_encodings, train_labels)
eval_dataset = TextClassifierDataset(eval_encodings, eval_labels)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    problem_type="multi_label_classification",
    num_labels=7
)

training_arguments = TrainingArguments(
    output_dir=".",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
)

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# Save just the model weights as a .pt file
torch.save(model.state_dict(), "action_model_weights.pt")
model.save_pretrained("action_saved_model")
tokenizer.save_pretrained("action_saved_model")
