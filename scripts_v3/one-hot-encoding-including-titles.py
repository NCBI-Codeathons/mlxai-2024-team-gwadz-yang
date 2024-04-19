import os
import pandas as pd
import pickle


DATA_DIR = '../data_v2/'
DATA_DIR_2 = '../data_v3/'

# the names of CDs and SuperFams have been sorted, duplication_removed, converted to lowercase
FILE_curated_cds = os.path.join(DATA_DIR, 'Curated_archCDs_v2_sorted')
FILE_curated_superfamilies = os.path.join(DATA_DIR, 'Curated_SuperFams_v2_sorted')
FILE_all_words_in_titles = os.path.join(DATA_DIR_2, 'words_counter.pkl')

# read words_counter.pkl
with open(FILE_all_words_in_titles, 'rb') as f:
    words_counter = pickle.load(f)


def main(file_path):

    # the lists of all CDs and superfamilies
    all_cds = []
    all_superfamilies = []
    all_title_words = list(words_counter.keys())


    # read FILE_curated_cds to populate the list of all CDs
    with open(FILE_curated_cds, 'r') as f:
        for line in f:
            all_cds.append(line.strip())


    # read FILE_curated_superfamilies to populate the list of all superfamilies
    with open(FILE_curated_superfamilies, 'r') as f:
        for line in f:
            all_superfamilies.append(line.strip())

    # read FILE_curated_SPARCLE_data into a dataframe
    # df = pd.read_csv(FILE_curated_SPARCLE_data, usecols=['CurName_simplified', 'SpecificArch', 'superfamilyarch'])
    df = pd.read_csv(file_path, usecols=['CurName', 'SpecificArch', 'superfamilyarch', 'TitleStrings'])

    output_file = os.path.join(DATA_DIR_2, 'Matrix_CurName_SpecifiedArchs_SuperFams_Titles_one_hot_encoding.tsv')

    # open the output file for writing
    with open(output_file, 'w') as f:
        # write the header
        # f.write('CurName_simplified\t')
        f.write('CurName\t')
        to_write = '\t'.join(all_cds)
        f.write(to_write)
        f.write('\t')
        to_write = '\t'.join(all_superfamilies)
        f.write(to_write)
        f.write('\t')
        to_write = '\t'.join(all_title_words)
        f.write(to_write)
        f.write('\n')

        # iterate over the rows of the dataframe
        for index, row in df.iterrows():
            # get the values of the columns
            cur_name = f"{row['CurName']}"
            f.write(cur_name)
            f.write('\t')

            specific_arch = row['SpecificArch']
            super_fams = row['superfamilyarch']

            arr = [0] * len(all_cds)
            if not pd.isna(specific_arch):
                for cd in specific_arch.split(' '):
                    if cd in all_cds:
                        arr[all_cds.index(cd)] = 1

            f.write('\t'.join([str(x) for x in arr]))
            f.write('\t')

            arr = [0] * len(all_superfamilies)
            if not pd.isna(super_fams):
                for sf in super_fams.split(' '):
                    if sf in all_superfamilies:
                        arr[all_superfamilies.index(sf)] = 1

            f.write('\t'.join([str(x) for x in arr]))
            f.write('\t')


            arr = [0] * len(all_title_words)
            if not pd.isna(row['TitleStrings']):
                titles = row['TitleStrings'].split('|')
                words = [word for title in titles for word in title.split()]
                for word in words:
                    if word in all_title_words:
                        arr[all_title_words.index(word)] = 1

            f.write('\t'.join([str(x) for x in arr]))
            f.write('\n')

    print('Done! The output file is:', output_file)
    print('The number of rows in the output file is:', len(df))

    print('\n')
    print("""
    # The output file is a matrix with the following columns:
    # CurName_simplified: the name of the protein
    # The next columns are binary values indicating the presence of a CD in the SpecificArch
    # The next columns are binary values indicating the presence of a SuperFam in the superfamilyarch
    # The next columns are binary values indicating the presence of a word in the title strings
    """)

if __name__ == '__main__':
    FILE_curated_SPARCLE_data = os.path.join(DATA_DIR, 'CuratedArch_simplifiedNames_culled.csv')
    # FILE_curated_SPARCLE_data = os.path.join(DATA_DIR, 'CuratedArch_simplifiedNames_titles_counts_v4.csv')
    # FILE_curated_SPARCLE_data = os.path.join(DATA_DIR, 'UnCuratedArchs_superfams__titles_v4.csv')
    main(FILE_curated_SPARCLE_data)
