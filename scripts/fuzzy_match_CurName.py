## This is code that attempts to reduce the complexity of the "CurName" (curated name) in the curated SPARCLE dataset by grouping the values based on similarity using the Levenshtein distance.

# !{sys.executable} -m pip install scikit-learn
# !{sys.executable} -m pip install fuzzywuzzy
# !{sys.executable} -m pip install python-Levenshtein
# import sklearn
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

import sys, os, json
import fuzzywuzzy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd
from pathlib import Path

local_dir = "/Users/christopher/Documents/Notes/misc/codeathon"
os.chdir(local_dir) 

git_dir = "/Users/christopher/Documents/GitHub/NCBI-Codeathons/mlxai-2024-team-gwadz-yang"
sfile = "{}/data/SPARCLE_IDS_curated.csv".format(git_dir)

df = pd.read_csv(sfile)

# Preprocessing
## For simplicity, lowercase the text

df['pCurName'] = df['CurName'].apply(lambda x: x.lower())

cn = list(set(df.CurName)) # 14947
pcn = list(set(df.pCurName)) # 14916


def group_strings(strings, threshold=90):
    """
    Returns a dictionary to store groups of similar strings. 
    This approach groups similar strings together using the fuzz.ratio() function. 
    You can adjust the similarity threshold to adjust stringency of grouping.
    """
    # if "CurName_groups_{}_threshold.json".format(threshold) file already exists, don't do this loop, instead read in groups:
    #
    #

    
    groups_filename = "CurName_groups_{}_threshold.json".format(threshold)
    groups_file = Path(groups_filename)
    if groups_file.is_file():
        print("Groups file already exists: {}".format(groups_file))
        with open(groups_file) as json_file:
            groups = json.load(json_file)
    else:

        groups = {}

        for i in range(0,len(strings)):
            string = strings[i]
            i+=1
            print("{}/{}: {}".format(i,len(strings),string))
            # Search for similar strings in existing groups
            matched_group = None
            for group in groups.values():
                for existing_string in group:
                    if fuzz.ratio(string, existing_string) > threshold:  # Adjust threshold as needed
                        matched_group = group
                        break
                if matched_group:
                    break
            
            # If a similar group is found, add string to that group
            if matched_group:
                matched_group.append(string)
            else:
                # Otherwise, create a new group
                groups[string] = [string]

        # export groups
        with open(groups_filename, 'w') as json_file:
            json.dump(groups, json_file)


    ## Normalize names
    # add some code for picking the curated name from each group?

    # Map each string to its corresponding group
    string_groups = {}
    items = list(groups.items())
    for i in range(0,len(items)):
        group,values = items[i]
        i+=1
        print("{}/{}: {}".format(i,len(groups.items()),group))
        for value in values:
            string_groups[value] = group
    
    return string_groups, groups

# run the string fuzzy grouping function on the pre-processed (lowercase) CurName column:
threshold=90
curated_names = df['pCurName']
string_groups, groups = group_strings(curated_names,threshold=threshold)

# Create a new column, gCurName, in the DataFrame with the grouped strings
df['gCurName'] = df['pCurName'].map(string_groups)

df.sort_values(by="pCurName",inplace=True)

out_file = "grouped_SPARCLE_IDS_curated_{}.csv".format(threshold)
df.to_csv(out_file,index=False)



