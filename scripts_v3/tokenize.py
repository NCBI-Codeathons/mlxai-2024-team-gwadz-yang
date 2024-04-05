from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./pretrained_model")
model = AutoModelForMaskedLM.from_pretrained("./pretrained_model")

# Define the input text
input_text = "This is a sample text to be tokenized."

# Tokenize the input text
tokenized_input = tokenizer(input_text, return_tensors="pt")

print('tokenized_input:', tokenized_input)

# Inspect the tokenized input
print('decoded:', tokenizer.decode(tokenized_input.input_ids[0]))

# Retrieve the vocabulary
vocab = tokenizer.get_vocab()
print(f"Vocabulary size: {len(vocab)}")