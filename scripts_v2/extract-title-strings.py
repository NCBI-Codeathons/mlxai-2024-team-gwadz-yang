import os
import pandas as pd


DATA_DIR = '../data_v2/'
FILE_curated_SPARCLE_data = os.path.join(DATA_DIR, 'CuratedArch_simplifiedNames_culled.csv')

# read FILE_curated_SPARCLE_data into a dataframe
df = pd.read_csv(FILE_curated_SPARCLE_data, usecols=['TitleStrings'])

# write the title strings to a file
output_file = os.path.join(DATA_DIR, 'TitleStrings')
with open(output_file, 'w') as f:
    for title in df['TitleStrings']:
        if not pd.isna(title):
            f.write(title + '\n')

print('Done')
