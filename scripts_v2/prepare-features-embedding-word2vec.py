import os
from gensim.models import Word2Vec
import numpy as np
import pandas as pd

DATA_DIR = '../data_v2/'
FILE_curated_SPARCLE_data = os.path.join(DATA_DIR, 'CuratedArch_simplifiedNames_culled.csv')

# dataset = [
#     {
#         "cur_name": "40s ribosomal protein s11",
#         "domain_ids": "cd06173 cl41468",
#         "superfamilies": "MFS MSCRAMM_ClfB",
#         "title": "Macrolide efflux protein A and similar proteins of the Major Facilitator Superfamily of transporters | MSCRAMM_ClfB"
#     },
#     # Add more data entries
# ]

dataset = []

df = pd.read_csv(FILE_curated_SPARCLE_data,
                 usecols=['CurName_simplified', 'SpecificArch', 'superfamilyarch', 'TitleStrings'])

for index, row in df.iterrows():
    cur_name = '' if pd.isna(row['CurName_simplified']) else row['CurName_simplified']

    dataset.append({
        "cur_name": '' if pd.isna(row['CurName_simplified']) else row['CurName_simplified'],
        "domain_ids": '' if pd.isna(row['SpecificArch']) else row['SpecificArch'],
        "superfamilies": '' if pd.isna(row['superfamilyarch']) else row['superfamilyarch'],
        "title": '' if pd.isna(row['TitleStrings']) else row['TitleStrings']
    })

# Set the parameters for the Word2Vec model
embedding_size = 100  # Dimensionality of the embedding vectors
window_size = 5  # Context window size
min_count = 1  # Minimum word frequency threshold
workers = 4  # Number of worker threads

# Train the Word2Vec model on the entire corpus
# Combine all tokens from domain IDs, superfamilies, and title
corpus = []
for record in dataset:
    cur_name_tokens = record["cur_name"].split()
    domain_ids_tokens = record["domain_ids"].split()
    superfamilies_tokens = record["superfamilies"].split()
    title_tokens = record["title"].split()
    corpus.append(cur_name_tokens + domain_ids_tokens + superfamilies_tokens + title_tokens)

model = Word2Vec(corpus, vector_size=embedding_size, window=window_size, min_count=min_count, workers=workers)

# Set the weights for each component
domain_ids_weight = 0.3
superfamilies_weight = 0.4
title_weight = 0.3

# Get the embeddings for all entries
embeddings = []

for record in dataset:
    domain_ids_tokens = record["domain_ids"].split()
    superfamilies_tokens = record["superfamilies"].split()
    title_tokens = record["title"].split()
    domain_ids_embeddings = [model.wv[token] for token in domain_ids_tokens if token in model.wv]
    superfamilies_embeddings = [model.wv[token] for token in superfamilies_tokens if token in model.wv]
    title_embeddings = [model.wv[token] for token in title_tokens if token in model.wv]

    domain_ids_embedding = np.mean(domain_ids_embeddings, axis=0) if domain_ids_embeddings else np.zeros(embedding_size)
    superfamilies_embedding = np.mean(superfamilies_embeddings, axis=0) if superfamilies_embeddings else np.zeros(
        embedding_size)
    title_embedding = np.mean(title_embeddings, axis=0) if title_embeddings else np.zeros(embedding_size)

    weighted_embedding = (
            domain_ids_weight * domain_ids_embedding +
            superfamilies_weight * superfamilies_embedding +
            title_weight * title_embedding
    )

    embeddings.append(weighted_embedding)

# Convert the embeddings to a numpy array
# embeddings = np.array(embeddings)

# print(f"Embeddings shape: {embeddings.shape}")
# print(embeddings[:3])
#
# similar_words = model.wv.most_similar('cd09869')
# print('Similar words to cd09869:', similar_words)
#
# similar_words = model.wv.most_similar('cd00165')
# print('Similar words to cd00165:', similar_words)
#
# similar_words = model.wv.most_similar('cd06173')
# print('Similar words to cd06173:', similar_words)
#
# similar_words = model.wv.most_similar('cd02553')
# print('Similar words to cd02553:', similar_words)

print(f"Embeddings shape: {len(embeddings)}")
print(embeddings[:3])

# Save the embeddings to a file
df_to_save = pd.DataFrame({'CurName': df['CurName_simplified'], 'features': embeddings})
df_to_save.to_pickle(os.path.join(DATA_DIR, 'Dataframe_CurName_features_embedding_word2vec.pkl'))
