import os
import pandas as pd
import sentencepiece as spm

DATA_DIR = '../data_v2/'

# the names of CDs and SuperFams have been sorted, duplication_removed, converted to lowercase
FILE_curated_cds = os.path.join(DATA_DIR, 'Curated_archCDs_v2_sorted')
FILE_curated_superfamilies = os.path.join(DATA_DIR, 'Curated_SuperFams_v2_sorted')
FILE_curated_SPARCLE_data = os.path.join(DATA_DIR, 'CuratedArch_simplifiedNames_culled.csv')
FILE_uncurated_SPARCLE_data = os.path.join(DATA_DIR, 'UnCuratedArchs_culled.csv')

# the dictionary of all CDs and SuperFams
all_cds_dict = {}
all_superfamilies_dict = {}

# read FILE_curated_cds to populate the list of all CDs
with open(FILE_curated_cds, 'r') as f:
    i = 1
    for line in f:
        all_cds_dict[line.strip()] = i
        i += 1

# read FILE_curated_superfamilies to populate the list of all superfamilies
with open(FILE_curated_superfamilies, 'r') as f:
    i = 1
    for line in f:
        all_superfamilies_dict[line.strip()] = i
        i += 1

# load the sentencepiece model
sp = spm.SentencePieceProcessor(model_file='titleStrings.model')


# read FILE_curated_SPARCLE_data into a dataframe
# df = pd.read_csv(FILE_uncurated_SPARCLE_data, usecols=['SpecificArch', 'superfamilyarch'])
# # output_file = os.path.join(DATA_DIR, 'Matrix_CurName_simplified_SpecifiedArchs_SuperFams_TitleStrings_label_encoding.pickle')
#
# max_len = 0
# i = 0
# for cds in df['SpecificArch']:
#     i += 1
#     if not pd.isna(cds):
#         n = len(cds.split())
#         if max_len < n:
#             max_len = n
#         if n == 41:
#             print(cds)
#
# print('max number of CDs in a row:', max_len, 'row:', i)
#
# max_len = 0
# i = 0
# for superfams in df['superfamilyarch']:
#     i += 1
#     if not pd.isna(superfams):
#         n = len(superfams.split())
#         if max_len < n:
#             max_len = n
#         if n == 32:
#             print(superfams)
#
# print('max number of superfamilies in a row:', max_len, 'row:', i)

# max_len = 0
#
# for titles in df['TitleStrings']:
#     if not pd.isna(titles):
#         a = sp.encode(titles)
#         if max_len < len(a):
#             max_len = len(a)
#
# print('max number of tokens in titles in a row:', max_len)


def encode_data(df):
    features = []
    for index, row in df.iterrows():
        # SpecificArch, max number of CDs in a row: 41
        a = [0] * 41
        if not pd.isna(row['SpecificArch']):
            for i, cds in enumerate(row['SpecificArch'].lower().split()):
                a[i] = all_cds_dict.get(cds, 0)

        # superfamilyarch, max number of superfamilies in a row: 32
        b = [0] * 32
        if not pd.isna(row['superfamilyarch']):
            for i, superfams in enumerate(row['superfamilyarch'].lower().split()):
                b[i] = all_superfamilies_dict.get(superfams, 0)

        # TitleStrings, max number of tokens in titles in a row: 340
        # c = [0] * 360
        # if not pd.isna(row['TitleStrings']):
        #     t = sp.encode(row['TitleStrings'])
        #     c[0:len(t)] = t

        # features.append(a + b + c)

        features.append(a + b)

    return features


def encode_training_data(input_file, output_file):
    df_ = pd.read_csv(input_file, usecols=['CurName_simplified', 'SpecificArch', 'superfamilyarch'])
    features = encode_data(df_)

    df_to_save = pd.DataFrame({'CurName_simplified': df_['CurName_simplified'], 'features': features})
    print(df_to_save.head())
    df_to_save.to_pickle(output_file)


def encode_testing_data(input_file, output_file):
    df_ = pd.read_csv(input_file, usecols=['CurName_simplified', 'SpecificArch', 'superfamilyarch'])
    features = encode_data(df_)

    df_to_save = pd.DataFrame({'CurName_simplified': df_['CurName_simplified'], 'features': features})
    print(df_to_save.head())
    df_to_save.to_pickle(output_file)


if __name__ == '__main__':
    FILE_curated_SPARCLE_data = os.path.join(DATA_DIR, 'CuratedArch_simplifiedNames_titles_counts_v4.csv')
    encode_training_data(FILE_curated_SPARCLE_data,
                         os.path.join(DATA_DIR,
                                      'DataFrame_Curated_CurName_SpecificArch_SuperFams_label_encoding.pickle'))


    # encode_testing_data(FILE_uncurated_SPARCLE_data,
    #                     os.path.join(DATA_DIR, 'DataFrame_UnCurated_SpecificArch_SuperFam_label_encoding.pickle'))
