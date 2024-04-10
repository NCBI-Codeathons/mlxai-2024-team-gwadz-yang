from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import os
import pickle


DATA_DIR = '../data_v2'
FILE_curated_SPARCLE_data = os.path.join(DATA_DIR, 'CuratedArch_simplifiedNames_culled.csv')
RESULTS_DIR = '../data_v3'


# Load your data into a pandas DataFrame
df = pd.read_csv(FILE_curated_SPARCLE_data, usecols=['CurName', 'TitleStrings'])
print(df.shape)
df.dropna(subset=['TitleStrings'], inplace=True)
print(df.shape)
# remove rows with empty title strings
df = df[df['TitleStrings'].str.strip() != '']
print(df.shape)

def remove_NA(row):
    names = [name.strip() for name in row.split('|')]
    names = [name for name in names if name != 'NA']
    return ' | '.join(names)

df['TitleStrings'] = df['TitleStrings'].apply(remove_NA)

# testing
# df = df[:5]

# convert the 'TitleStrings' column to a list of strings
# titleStrings = df['TitleStrings'].tolist()

# index_longest = df['TitleStrings'].str.len().idxmax()
#
# # Retrieve the row with the longest string
# row_longest = df.loc[index_longest]
#
# # Print the row
# print(row_longest['TitleStrings'])
# print("Length of the string in 'titleStrings' column:", len(row_longest['TitleStrings']))


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# Load AutoModel from huggingface model repository
# tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
# model = AutoModel.from_pretrained("microsoft/biogpt")

tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
model = AutoModel.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")

# Compute token embeddings
embeddings = []

# Process data row by row
for index, row in df.iterrows():
    # Tokenize sentence
    encoded_input = tokenizer(
        row['TitleStrings'], padding=True, truncation=True, max_length=512, return_tensors="pt"
    )

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform mean pooling
    sentence_embedding = mean_pooling(model_output, encoded_input["attention_mask"])

    # Append embedding to list
    embeddings.append(sentence_embedding.flatten().numpy())


df = pd.DataFrame({'CurName': df['CurName'], 'features': embeddings})
print(df.head(), df.shape)

# Save the DataFrame
# output_file = os.path.join(DATA_DIR, 'Dataframe_CurName_features_embedding_biogpt.pkl')
output_file = os.path.join(RESULTS_DIR, 'Dataframe_CurName_features_embedding_biomedbert.pkl')
df.to_pickle(output_file)
