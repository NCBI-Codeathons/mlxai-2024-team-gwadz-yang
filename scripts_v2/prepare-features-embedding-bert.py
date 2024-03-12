import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import os

DATA_DIR = '../data_v2'
FILE_curated_SPARCLE_data = os.path.join(DATA_DIR, 'CuratedArch_simplifiedNames_culled.csv')


# Load your data into a pandas DataFrame
data = pd.read_csv(FILE_curated_SPARCLE_data, usecols=['CurName', 'SpecificArch', 'superfamilyarch', 'TitleStrings'])

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to tokenize and encode the text data
def encode_data(row):
    text = f"{row['SpecificArch']} <sep> {row['superfamilyarch']} <sep> {row['TitleStrings']}"
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    return encoded_input

# Tokenize and encode the data
encoded_data = data.apply(lambda row: encode_data(row), axis=1)

# print(type(encoded_data))  # <class 'pandas.core.series.Series'>

# Generate contextual embeddings
embeddings = []
with torch.no_grad():
    for encoded_input in encoded_data:
        output = model(**encoded_input)
        embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)

# save the embeddings as a DataFrame
df = pd.DataFrame({'CurName': data['CurName'], 'features': embeddings})
print(df.head())

# Save the DataFrame
output_file = os.path.join(DATA_DIR, 'Dataframe_CurName_features_embedding_bert.pkl')
df.to_pickle(output_file)




