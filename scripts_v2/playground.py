from transformers import BioGptTokenizer, BioGptForCausalLM
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
text = "Mismatch repair protein MLH1"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

predicted_token_ids = output.logits.argmax(dim=-1)

# Decode the predicted token IDs to text
predicted_text = tokenizer.decode(predicted_token_ids[0])

print("Input text:", text)
print("Generated text:", predicted_text)
