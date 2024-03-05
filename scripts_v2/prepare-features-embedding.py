from collections import defaultdict

# Domain ID sequences
domain_id_sequences = ["ID1 ID5 ID10", "ID3 ID7", "ID2 ID5 ID8 ID10"]  # Replace with your actual list of sequences

# Tokenize the sequences
tokenized_sequences = [seq.split() for seq in domain_id_sequences]

# Create a vocabulary
vocabulary = defaultdict(lambda: len(vocabulary) + 1)
for seq in tokenized_sequences:
    for domain_id in seq:
        vocabulary[domain_id]

# Convert sequences to integer IDs
corpus = [[vocabulary[domain_id] for domain_id in seq] for seq in tokenized_sequences]

# Print the vocabulary and the corpus
print("Vocabulary:", vocabulary)
print("Corpus:", corpus)
