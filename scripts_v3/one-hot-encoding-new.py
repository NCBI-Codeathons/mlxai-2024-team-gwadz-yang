import os
import pandas as pd


def create_one_hot_encoding_file(input_file, cols=['CurName', 'Superfamilies', 'SpecificArch']):
    # read the input file
    df = pd.read_csv(input_file, usecols=cols, sep='\t')
    all_cds = set()
    all_superfamilies = set()

    # populate the list of all CDs and superfamilies
    for index, row in df.iterrows():
        specific_arch = row[cols[2]]
        super_fams = row[cols[1]]

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
    output_dir = os.path.dirname(input_file)
    output_file = os.path.join(output_dir, 'Matrix_one_hot_encoding.tsv')

    # open the output file for writing
    with open(output_file, 'w') as f:
        f.write(cols[0] + '\t')
        to_write = '\t'.join(all_cds)
        f.write(to_write)
        f.write('\t')
        to_write = '\t'.join(all_superfamilies)
        f.write(to_write)
        f.write('\n')

        # iterate over the rows of the dataframe
        for index, row in df.iterrows():
            cur_name = f"{row[cols[0]]}"
            f.write(cur_name)
            f.write('\t')

            specific_arch = row[cols[2]]
            super_fams = row[cols[1]]

            arr = [0] * len(all_cds)
            if not pd.isna(specific_arch):
                for cd in specific_arch.split(' '):
                    if cd in all_cds:
                        arr[all_cds.index(cd)] = 1

            to_write = '\t'.join(map(str, arr))
            f.write(to_write)
            f.write('\t')

            arr = [0] * len(all_superfamilies)
            if not pd.isna(super_fams):
                for sf in super_fams.split(' '):
                    if sf in all_superfamilies:
                        arr[all_superfamilies.index(sf)] = 1

            to_write = '\t'.join(map(str, arr))
            f.write(to_write)
            f.write('\n')

    print(f'One-hot encoding file created: {output_file}')


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('Usage: python one-hot-encoding-new.py <input_file>')
        sys.exit(1)

    input_file = sys.argv[1]
    # validate the input file
    if not os.path.exists(input_file):
        print(f'File not found: {input_file}')
        sys.exit(1)

    # create the one-hot encoding file
    create_one_hot_encoding_file(input_file)
