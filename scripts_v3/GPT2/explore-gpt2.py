from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from datasets import Dataset
import torch

# Read the CSV file into a pandas DataFrame
File_path = '../curName_specificArch_superfamilyArch.csv'
with open(File_path, 'r') as f:
    texts = [line.strip() for line in f.readlines()]

print(texts[:5])
# Split the data into training and validation sets
train_texts, val_texts = train_test_split(texts, test_size=0.2, random_state=42)

# Load the pretrained GPT tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to the end-of-sequence token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenize and encode the domain-specific corpus
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Convert encodings to datasets
train_dataset = Dataset.from_dict(train_encodings)
val_dataset = Dataset.from_dict(val_encodings)

# Function to process the batch
def process_batch(batch):
    inputs = torch.tensor(batch["input_ids"])
    labels = inputs.clone()

    batch["input_ids"] = inputs.tolist()
    batch["labels"] = labels.tolist()
    return batch

train_dataset = train_dataset.map(process_batch, batched=True, batch_size=len(train_dataset))
val_dataset = val_dataset.map(process_batch, batched=True, batch_size=len(val_dataset))

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
)

# Set up the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Pretrain the model
trainer.train()

# Save the pretrained model
trainer.save_model("./pretrained_model")