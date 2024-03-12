from gensim.models import KeyedVectors
import os
import numpy as np

File_bio_embedding_extrinsic = os.path.join('../data_v2', 'bio_embedding_extrinsic')

# Load the pre-trained BioWordVec embeddings
model = KeyedVectors.load_word2vec_format(File_bio_embedding_extrinsic, binary=True)

# Your output strings
output_strings = [
    "bifunctional pyridoxal-dependent decarboxylase/aldehyde reductase",
    "crp/fnr family transcriptional regulator",
    "ctbp domain-containing protein",
    "cytoplasmic phosphatidylinositol transfer protein 1",
    "damage-control phosphatase armt1 family protein",
    "dnaj-like protein c11 c-terminal domain-containing protein",
    "duf4097 domain-containing protein",
    "duf4411 family protein",
    "exonuclease",
    "filamin/abp280 repeat-containing protein",
    "fpg/nei family dna glycosylase",
    "fyve zinc finger domain-containing protein",
    "gen family endonuclease",
]

output_vectors = []
for output_string in output_strings:
    tokens = output_string.split()  # Split the string into tokens based on whitespace
    token_vectors = []
    for token in tokens:
        if token in model.key_to_index:
            token_vector = model[token]
            token_vectors.append(token_vector)
    if token_vectors:
        output_vector = np.mean(token_vectors, axis=0)
        output_vectors.append(output_vector)
    else:
        output_vectors.append(np.zeros(model.vector_size))

# Print the dimensions of the output vectors
print("Output Vectors Shape:", np.array(output_vectors).shape)
print(output_vectors)