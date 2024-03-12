from transformers import BioGptTokenizer
import torch

# Load the BioGptTokenizer
tokenizer = BioGptTokenizer.from_pretrained('microsoft/biogpt')

# Your protein names
protein_names = [
    "bifunctional pyridoxal-dependent decarboxylase/aldehyde reductase",
    "crp/fnr family transcriptional regulator",
    "ctbp domain-containing protein",
    "cytoplasmic phosphatidylinositol transfer protein 1",
    "damage-control phosphatase armt1 family protein",
    "dnaj-like protein c11 c-terminal domain-containing protein",
    "duf4097 domain-containing protein",
    "duf4411 family protein",
    "exonuclease",
    "filamin/abp280 repeat-containing protein",
    "fpg/nei family dna glycosylase",
    "fyve zinc finger domain-containing protein",
    "gen family endonuclease",
]

# Tokenize each protein name
tokenized_protein_names = []
for protein_name in protein_names:
    tokens = tokenizer.tokenize(protein_name)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    tokenized_protein_names.append(token_ids)

# Print the tokenized protein names
for i, token_ids in enumerate(tokenized_protein_names):
    print(f"Protein {i+1}: {token_ids}")

# Output:
print(tokenized_protein_names)

print(tokenizer.decode([25181, 0, 0]))