import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Tokenize input text using a fine-tuned BiomedBERT model.')
parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned BiomedBERT model')
args = parser.parse_args()

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForMaskedLM.from_pretrained(args.model_path)

# Define the input text
input_text = "This is a sample text to be tokenized."

# Tokenize the input text
tokenized_input = tokenizer(input_text, return_tensors="pt")
print('Tokenized input:', tokenized_input)

# Inspect the tokenized input
print(tokenizer.decode(tokenized_input.input_ids[0]))
print('decoded:', tokenizer.decode(tokenized_input.input_ids[0]))

# Retrieve the vocabulary
vocab = tokenizer.get_vocab()
print(f"Vocabulary size: {len(vocab)}")