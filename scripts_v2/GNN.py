import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np

# Preprocessing functions
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

# Word2Vec embedding functions
def create_word2vec_embeddings(domains, embedding_size=100):
    short_names = [preprocess_text(domain['shortname']) for domain in domains]
    titles = [preprocess_text(domain['title']) for domain in domains]
    descriptions = [preprocess_text(domain['description']) for domain in domains]

    texts = short_names + titles + descriptions

    model = Word2Vec(texts, vector_size=embedding_size, window=5, min_count=1, workers=4)

    word2vec_embeddings = {}
    for domain in domains:
        accession = domain['accession']
        short_name = preprocess_text(domain['shortname'])
        title = preprocess_text(domain['title'])
        description = preprocess_text(domain['description'])

        short_name_embedding = np.mean([model.wv[word] for word in short_name if word in model.wv], axis=0)
        title_embedding = np.mean([model.wv[word] for word in title if word in model.wv], axis=0)
        description_embedding = np.mean([model.wv[word] for word in description if word in model.wv], axis=0)

        word2vec_embeddings[accession] = np.concatenate((short_name_embedding, title_embedding, description_embedding))

    return word2vec_embeddings

# BERT embedding functions
def create_bert_embeddings(domains, model_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    bert_embeddings = {}
    for domain in domains:
        accession = domain['accession']
        description = domain['description']
        short_name = domain['shortname']
        title = domain['title']

        text = f"{accession} [SEP] {short_name} [SEP] {title} [SEP] {description}"

        encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            output = model(**encoded_input)
            embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()

        bert_embeddings[accession] = embedding

    return bert_embeddings

# Graph creation function
def create_domain_graph(domains):
    G = nx.DiGraph()

    for domain in domains:
        G.add_node(domain['accession'],
                   shortname=domain['shortname'],
                   title=domain['title'],
                   description=domain['description'])

    for domain in domains:
        if 'parent' in domain:
            G.add_edge(domain['parent'], domain['accession'])

    return G

# GNN model
class DomainGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DomainGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Main code
domains = [
    {'accession': 'PF00001', 'shortname': '7tm_1', 'title': '7 transmembrane receptor (rhodopsin family)', 'description': 'This domain represents the...'},
    {'accession': 'PF00002', 'shortname': '7tm_2', 'title': '7 transmembrane receptor (Secretin family)', 'description': 'This domain represents the...', 'parent': 'PF00001'},
    {'accession': 'PF00003', 'shortname': '7tm_3', 'title': '7 transmembrane receptor (Metabotropic glutamate family)', 'description': 'This domain represents the...', 'parent': 'PF00001'},
    # ... more domain dictionaries ...
]

# Create embeddings
word2vec_embeddings = create_word2vec_embeddings(domains)
# bert_embeddings = create_bert_embeddings(domains)

# Create feature matrix
feature_matrix = []
for domain in domains:
    accession = domain['accession']
    word2vec_embedding = word2vec_embeddings[accession]
    # bert_embedding = bert_embeddings[accession]
    feature_matrix.append(np.concatenate((word2vec_embedding, [])))

feature_matrix = torch.tensor(feature_matrix, dtype=torch.float)

# Create graph
domain_graph = create_domain_graph(domains)

# Create edge index
edge_index = nx.to_edgelist(domain_graph)
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Create GNN model
input_dim = feature_matrix.shape[1]
hidden_dim = 64
output_dim = 32
model = DomainGNN(input_dim, hidden_dim, output_dim)

# Train the GNN
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(feature_matrix, edge_index)
    # Define your loss function and compute the loss
    loss = ...
    loss.backward()
    optimizer.step()


# Generate embeddings for CD accessions or short names
model.eval()
with torch.no_grad():
    node_embeddings = model(feature_matrix, edge_index)

# Create a dictionary to map CD accessions to their embeddings
accession_to_embedding = {}
for i, domain in enumerate(domains):
    accession = domain['accession']
    embedding = node_embeddings[i].numpy()
    accession_to_embedding[accession] = embedding

# Create a dictionary to map short names to their embeddings
shortname_to_embedding = {}
for i, domain in enumerate(domains):
    shortname = domain['shortname']
    embedding = node_embeddings[i].numpy()
    shortname_to_embedding[shortname] = embedding

# Function to retrieve embedding for a given CD accession
def get_embedding_by_accession(accession):
    if accession in accession_to_embedding:
        return accession_to_embedding[accession]
    else:
        return None

# Function to retrieve embedding for a given short name
def get_embedding_by_shortname(shortname):
    if shortname in shortname_to_embedding:
        return shortname_to_embedding[shortname]
    else:
        return None