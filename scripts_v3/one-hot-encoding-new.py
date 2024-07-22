import os
import pandas as pd

def write_to_file(f, ALL_CDS, ALL_SUPERFAMS, COLUMNS, DF):
    f.write(COLUMNS[0] + '\t')
    to_write = '\t'.join(ALL_CDS)
    f.write(to_write)
    f.write('\t')
    to_write = '\t'.join(ALL_SUPERFAMS)
    f.write(to_write)
    f.write('\n')

    for index, row in DF.iterrows():
        f.write(f"{row[COLUMNS[0]]}")
        f.write('\t')

        specific_arch = row[COLUMNS[1]]
        super_fams = row[COLUMNS[2]]

        arr = [0] * len(ALL_CDS)
        if not pd.isna(specific_arch):
            for cd in specific_arch.split(' '):
                if cd in ALL_CDS:
                    arr[ALL_CDS.index(cd)] = 1

        to_write = '\t'.join(map(str, arr))
        f.write(to_write)
        f.write('\t')

        arr = [0] * len(ALL_SUPERFAMS)
        if not pd.isna(super_fams):
            for sf in super_fams.split(' '):
                if sf in ALL_SUPERFAMS:
                    arr[ALL_SUPERFAMS.index(sf)] = 1

        to_write = '\t'.join(map(str, arr))
        f.write(to_write)
        f.write('\n')


def create_one_hot_encoding_file(curated_file, uncurated_file):
    # read the input file
    cols = ['CurName', 'SpecificArch', 'Superfamilies']
    df = pd.read_csv(curated_file, usecols=cols, sep='\t')
    all_cds = set()
    all_superfamilies = set()

    # populate the list of all CDs and superfamilies
    for index, row in df.iterrows():
        specific_arch = row[cols[1]]
        super_fams = row[cols[2]]

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

    # get the directory of the input file
    output_dir = os.path.dirname(curated_file)
    output_file = os.path.join(output_dir, 'curated_one_hot_encoding_matrix.tsv')

    with open(output_file, 'w') as f:
        write_to_file(f, all_cds, all_superfamilies, cols, df)

    print(f'One-hot encoding file for curated data was created: {output_file}')

    if uncurated_file:
        output_dir = os.path.dirname(uncurated_file)
        output_file = os.path.join(output_dir, 'uncurated_one_hot_encoding_matrix.tsv')

        cols = ['ArchId', 'SpecificArch', 'Superfamilies']
        df = pd.read_csv(uncurated_file, usecols=cols, sep='\t')

        with open(output_file, 'w') as f:
            write_to_file(f, all_cds, all_superfamilies, cols, df)

        print(f'One-hot encoding file for uncurated data was created: {output_file}')





if __name__ == '__main__':
    import sys
    import argparse

    # add arguments to the script
    parser = argparse.ArgumentParser(description='Create one-hot encoding file for curated and uncurated data')
    parser.add_argument('curated_file', help='The path to the curated file')
    parser.add_argument('--uncurated_file', help='The path to the uncurated file')

    # parse the arguments
    args = parser.parse_args()
    _curated_file = args.curated_file
    _uncurated_file = args.uncurated_file

    # Check if the curated file exists
    if not os.path.isfile(_curated_file):
        print(f"Error: The curated file '{_curated_file}' does not exist.")
        sys.exit(1)

    # If uncurated file is provided, check if it exists
    if _uncurated_file and not os.path.isfile(_uncurated_file):
        print(f"Error: The uncurated file '{_uncurated_file}' does not exist.")
        sys.exit(1)

    # create the one-hot encoding file
    create_one_hot_encoding_file(_curated_file, _uncurated_file)
