#!/usr/bin/env python3


import numpy as np
import pandas as pd
import os
import pickle
import sentencepiece as spm

DATA_DIR = "../data"
all_models_sorted_file = os.path.join(DATA_DIR, 'CD_models_curated_sorted')
all_curnames_file = os.path.join(DATA_DIR, 'simplified_Curname_set.csv')
all_superfamily_file = os.path.join(DATA_DIR, 'super_families_curated_sorted')
all_title_strings_file = os.path.join(DATA_DIR, 'Title_Strings_sorted_uniq')


# load indices dictionaries
with open(os.path.join(DATA_DIR, 'CD_models_indices.pickle'), 'rb') as f:
    domain_model_indices = pickle.load(f)


with open(os.path.join(DATA_DIR, 'super_families_indices.pickle'), 'rb') as f:
    superfamily_indices = pickle.load(f)


# --- Load the Trained Model ---
sp_titles = spm.SentencePieceProcessor(model_file='titles.model')
vocab_titles = {sp_titles.id_to_piece(i): i for i in range(sp_titles.get_piece_size())}

sp_curnames = spm.SentencePieceProcessor(model_file='curnames.model')
vocab_curnames = {sp_curnames.id_to_piece(i): i for i in range(sp_curnames.get_piece_size())}

# Function to convert a protein name into a target vector
def name_to_target_vector(name, sp, vocab):
  subwords = sp.encode(name, out_type=str)
  target_vector = [vocab.get(token, vocab['<unk>']) for token in subwords]
  return target_vector



def read_file_to_df(sparcle_data_file_path):
    # Specify the columns you want to read from the CSV file
    columns_to_read = ['ArchId', 'CurName', 'SpecificArch', 'superfamilyarch', 'TitleStrings']

    # Read specific columns of the CSV file into a DataFrame
    return pd.read_csv(sparcle_data_file_path, usecols=columns_to_read)

def generate_result(df):
    encoded_features = {}
    encoded_outputs = {}

    for index, row in df.iterrows():
        dms = row['SpecificArch']
        a1 = [0] * 30  # we use 30 positions to hold the indices of individual SpecificArch
        if not pd.isna(dms):
            arr = [domain_model_indices.get(m, 0) for m in dms.split(' ')]
            a1[:len(arr)] = arr

        sfa = row['superfamilyarch']
        a2 = [0] * 20  # we use 20 positions to hold the numeric values representing the tokenized superfamilyarch
        if not pd.isna(sfa):
            arr = [superfamily_indices.get(s, 0) for s in sfa.split(' ')]
            a2[:len(arr)] = arr

        title = row['TitleStrings']
        a3 = [0] * 400
        if not pd.isna(title):
            vec = name_to_target_vector(title.lower(), sp_titles, vocab_titles)
            a3[0:len(vec)] = vec

        curname = row['CurName']
        a4 = [0] * 60
        if not pd.isna(curname):
            vec = name_to_target_vector(curname, sp_curnames, vocab_curnames)
            a4[0:len(vec)] = vec

        if pd.isna(row['ArchId']):
            continue

        archid = str(int(row['ArchId']))
        encoded_features[archid] = a1 + a2 + a3
        encoded_outputs[archid] = a4


    with open(os.path.join(DATA_DIR, 'encoded_features_uncurated.pickle'), 'wb') as f:
        pickle.dump(encoded_features, f)

    with open(os.path.join(DATA_DIR, 'encoded_outputs_uncurated.pickle'), 'wb') as f:
        pickle.dump(encoded_outputs, f)





if __name__ == '__main__':
    data_uncurated_path = os.path.join(DATA_DIR, 'SPARCLE_IDS_UNcurated_TITLES_modTitleStrings.csv')
    generate_result(read_file_to_df(data_uncurated_path))
    print('Done')