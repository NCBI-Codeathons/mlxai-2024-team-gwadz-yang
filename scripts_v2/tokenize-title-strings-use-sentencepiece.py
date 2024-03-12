import os
import sentencepiece as spm

DATA_DIR = "../data_v2"
File_title_strings = os.path.join(DATA_DIR, 'TitleStrings')

# train sentencepiece model
spm.SentencePieceTrainer.train(
    input=File_title_strings,
    model_prefix='titleStrings',
    vocab_size=6566,
    character_coverage=0.9995
)

# load sentencepiece model
sp = spm.SentencePieceProcessor(model_file='titleStrings.model')


# encode some strings
print('Testing the encoding of some strings:')

names = [
    'Conjugate Transporter-2 (CT2) Family protein',
    'SprB repeat | Sodium:alanine symporter family',
    'Asp23 family, cell envelope-related function',
    'PH-like'
]

for name in names:
    print(name, sp.encode(name))


print('encode vs encode_as_pieces:')
print('encode:', sp.encode('Conjugate Transporter-2 (CT2) Family protein'))
print('encode_as_pieces:', sp.encode_as_pieces('Conjugate Transporter-2 (CT2) Family protein'))