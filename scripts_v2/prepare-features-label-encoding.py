import os
import pandas as pd
import sentencepiece as spm

DATA_DIR = '../data_v2/'

# the names of CDs and SuperFams have been sorted, duplication_removed, converted to lowercase
FILE_curated_cds = os.path.join(DATA_DIR, 'Curated_archCDs_v2_sorted')
FILE_curated_superfamilies = os.path.join(DATA_DIR, 'Curated_SuperFams_v2_sorted')
FILE_curated_SPARCLE_data = os.path.join(DATA_DIR, 'CuratedArch_simplifiedNames_culled.csv')

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
df = pd.read_csv(FILE_curated_SPARCLE_data, usecols=['CurName_simplified', 'SpecificArch', 'superfamilyarch', 'TitleStrings'])

output_file = os.path.join(DATA_DIR, 'Matrix_CurName_simplified_SpecifiedArchs_SuperFams_TitleStrings_label_encoding.pickle')

# max_len = 0
# for cds in df['SpecificArch']:
#     if not pd.isna(cds):
#         if max_len < len(cds.split(' ')):
#             max_len = len(cds.split(' '))
#
# print('max number of CDs in a row:', max_len)
#
# max_len = 0
# for superfams in df['superfamilyarch']:
#     if not pd.isna(superfams):
#         if max_len < len(superfams.split(' ')):
#             max_len = len(superfams.split(' '))
#
# print('max number of superfamilies in a row:', max_len)
#
# max_len = 0
#
# for titles in df['TitleStrings']:
#     if not pd.isna(titles):
#         a = sp.encode(titles)
#         if max_len < len(a):
#             max_len = len(a)
#
# print('max number of tokens in titles in a row:', max_len)

# label encoding
curNames = []
features = []

for index, row in df.iterrows():
    curNames.append(row['CurName_simplified'])

    # SpecificArch, max number of CDs in a row: 20
    a = [0] * 24
    if not pd.isna(row['SpecificArch']):
        for i, cds in enumerate(row['SpecificArch'].lower().split(' ')):
            a[i] = all_cds_dict[cds]

    # superfamilyarch, max number of superfamilies in a row: 9
    b = [0] * 12
    if not pd.isna(row['superfamilyarch']):
        for i, superfams in enumerate(row['superfamilyarch'].lower().split(' ')):
            b[i] = all_superfamilies_dict[superfams]

    # TitleStrings, max number of tokens in titles in a row: 340
    c = [0] * 360
    if not pd.isna(row['TitleStrings']):
        c = sp.encode(row['TitleStrings'])

    features.append(a + b + c)


df = pd.DataFrame({'CurName_simplified': curNames, 'features': features})
print(df.head())
df.to_pickle(output_file)
print('label encoded data saved to', output_file)