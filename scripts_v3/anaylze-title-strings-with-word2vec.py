import os
import pandas as pd
import pickle


DATA_DIR = '../data_v2'
FILE_curated_SPARCLE_data = os.path.join(DATA_DIR, 'CuratedArch_simplifiedNames_culled.csv')
FILE_uncurated_SPARCLE_data = os.path.join(DATA_DIR, 'UnCuratedArchs_superfams__titles_v4.csv')
RESULTS_DIR = '../data_v3'

# Load your data into a pandas DataFrame
df1 = pd.read_csv(FILE_curated_SPARCLE_data, usecols=['TitleStrings'])
print(df1.shape)
df2 = pd.read_csv(FILE_uncurated_SPARCLE_data, usecols=['TitleStrings'])
print(df2.shape)

# collect all title strings
df = pd.concat([df1, df2], axis=0, ignore_index=True)
print(df.shape)

# remove rows with empty title strings
df.dropna(subset=['TitleStrings'], inplace=True)
print(df.shape)

words_counter = {}
all_titles = []


for row in df['TitleStrings']:
    # remove 'NA' from the title strings
    titles = [title.strip() for title in row.split('|') if title.strip() != 'N/A']
    all_titles.extend(titles)
    words = [word for title in titles for word in title.split()]
    for word in words:
        if word in words_counter:
            words_counter[word] += 1
        else:
            words_counter[word] = 1


print('count of word "biosynthesis":' ,words_counter['biosynthesis'])

# sort the words by frequency
sorted_words = sorted(words_counter.items(), key=lambda x: x[1], reverse=True)
print(len(sorted_words))
print(sorted_words[:10])


# save all titles to a file
with open(os.path.join(RESULTS_DIR, 'all_titles.txt'), 'w') as f:
    for title in all_titles:
        f.write(title)
        f.write('\n')

# save the words_counter to a pickle file
with open(os.path.join(RESULTS_DIR, 'words_counter.pkl'), 'wb') as f:
    pickle.dump(words_counter, f)


# analyze the title strings with word2vec

from gensim.models import Word2Vec


# Preprocessing function
def preprocess_title(text):
    text = text.lower()  # Lowercase
    return [word.strip() for word in text.split()]



# Preprocess titles
processed_titles = [preprocess_title(title) for title in all_titles]

# Define model parameters
vector_size = 100  # Dimensionality of word vectors
window_size = 5  # Context window size
min_count = 2  # Minimum word count

# Build the Word2Vec model
model = Word2Vec(processed_titles, sg=1, window=window_size, min_count=min_count, vector_size=vector_size)

# save the model
model.save(os.path.join(RESULTS_DIR, 'word2vec.model'))

# Example usage: Find similar words to "Learning"
similar_words = model.wv.most_similar(positive=["biosynthesis"], topn=3)
print(f"Similar words to 'biosynthesis': {similar_words}")

similar_words = model.wv.most_similar(positive=["transcriptional"], topn=3)
print(f"Similar words to 'transcriptional': {similar_words}")

# Example usage: Find the odd word out
odd_word = model.wv.doesnt_match(["dna", "rna", "protein", "carbohydrate", "cell"])
print(f"Odd word out: {odd_word}")

# Example usage: Calculate similarity between two words
similarity = model.wv.similarity("transcription", "transcriptional")
print(f"Similarity between 'transcription' and 'transcriptional': {similarity}")









