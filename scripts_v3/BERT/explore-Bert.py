from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForMaskedLM, TrainingArguments, Trainer
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
import torch

# Read the CSV file into a pandas DataFrame
File_path = './curName_specificArch_superfamilyArch.csv'
with open(File_path, 'r') as f:
    texts = [line.strip() for line in f.readlines()]

print(texts[:5])
# Split the data into training and validation sets
train_texts, val_texts = train_test_split(texts, test_size=0.2, random_state=42)

# Load the pretrained tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Tokenize and encode the domain-specific corpus
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_special_tokens_mask=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_special_tokens_mask=True)

# Convert encodings to datasets
train_dataset = Dataset.from_dict(train_encodings)
val_dataset = Dataset.from_dict(val_encodings)

# Function to mask tokens and generate labels
def mask_tokens(batch):
    inputs = torch.tensor(batch["input_ids"])
    special_tokens_mask = torch.tensor(batch["special_tokens_mask"], dtype=torch.bool)

    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, 0.15)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    batch["input_ids"] = inputs.tolist()
    batch["labels"] = labels.tolist()
    return batch

train_dataset = train_dataset.map(mask_tokens, batched=True, batch_size=len(train_dataset))
val_dataset = val_dataset.map(mask_tokens, batched=True, batch_size=len(val_dataset))

# Set up the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Set up the training arguments and trainer
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Pretrain the model
trainer.train()

# Save the pretrained model
trainer.save_model("./pretrained_model")