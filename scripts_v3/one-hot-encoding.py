import os
import pandas as pd
import pickle

DATA_DIR = '../data_v3/'

# the names of CDs and SuperFams have been sorted, duplication_removed, converted to lowercase
FILE_curated = os.path.join(DATA_DIR, 'CuratedArch_simplifiedNames_titles_counts_v4.txt')
FILE_uncurated = os.path.join(DATA_DIR, 'UnCuratedArchs_superfams__titles_v4.txt')


def main():
    # the lists of all CDs and superfamilies
    all_cds = set()
    all_superfamilies = set()

    # read FILE_curated to populate the list of all CDs
    df_1 = pd.read_csv(FILE_curated, usecols=['CurName_simplified', 'SpecificArch', 'superfamilyarch'], sep='\t')
    df_2 = pd.read_csv(FILE_uncurated, usecols=['ArchId', 'SpecificArch', 'superfamilyarch'], sep='\t')

    for index, row in df_1.iterrows():
        specific_arch = row['SpecificArch']
        super_fams = row['superfamilyarch']

        if not pd.isna(specific_arch):
            for cd in specific_arch.split(' '):
                all_cds.add(cd)

        if not pd.isna(super_fams):
            for sf in super_fams.split(' '):
                all_superfamilies.add(sf)

    all_cds = sorted(list(all_cds))
    all_superfamilies = sorted(list(all_superfamilies))

    print('The number of CDs:', len(all_cds), all_cds[:5])
    print('The number of SuperFams:', len(all_superfamilies), all_superfamilies[:5])


    output_file_1 = os.path.join(DATA_DIR, 'Matrix_CurName_SpecifiedArchs_SuperFams_one_hot_encoding.tsv')
    output_file_2 = os.path.join(DATA_DIR, 'Matrix_UnCurName_SpecifiedArchs_SuperFams_one_hot_encoding.tsv')

    # open the output file 1 for writing
    with open(output_file_1, 'w') as f:
        # write the header
        f.write('CurName_simplified\t')
        to_write = '\t'.join(all_cds)
        f.write(to_write)
        f.write('\t')
        to_write = '\t'.join(all_superfamilies)
        f.write(to_write)
        f.write('\n')

        # iterate over the rows of the dataframe
        for index, row in df_1.iterrows():
            # get the values of the columns
            cur_name = f"{row['CurName_simplified']}"
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
            f.write('\n')

    print('The output file is:', output_file_1)
    print('The number of rows in the output file is:', len(df_1))

    # open the output file 2 for writing
    with open(output_file_2, 'w') as f:
        # write the header
        f.write('ArchId\t')
        to_write = '\t'.join(all_cds)
        f.write(to_write)
        f.write('\t')
        to_write = '\t'.join(all_superfamilies)
        f.write(to_write)
        f.write('\n')

        # iterate over the rows of the dataframe
        for index, row in df_2.iterrows():
            # get the values of the columns
            archid = f"{row['ArchId']}"
            f.write(archid)
            f.write('\t')

            specific_arch = row['SpecificArch']
            super_fams = row['superfamilyarch']

            arr = [0] * len(all_cds)
            if not pd.isna(specific_arch):
                for cd in specific_arch.split():
                    if cd in all_cds:
                        arr[all_cds.index(cd)] = 1

            f.write('\t'.join([str(x) for x in arr]))
            f.write('\t')

            arr = [0] * len(all_superfamilies)
            if not pd.isna(super_fams):
                for sf in super_fams.split():
                    if sf in all_superfamilies:
                        arr[all_superfamilies.index(sf)] = 1

            f.write('\t'.join([str(x) for x in arr]))
            f.write('\n')

    print('The output file is:', output_file_2)
    print('The number of rows in the output file is:', len(df_2))


if __name__ == '__main__':
    main()
