# Team Project Name

List of participants and affiliations:
- Participant, Affiliation (Team Leader)
- Participant, Affiliation

## Project Goals
 SPARCLE, the Subfamily Protein Architecture Labeling Engine, is a resource for the functional characterization and labeling of protein sequences that have been grouped by their characteristic conserved domain architecture. A domain architecture is defined as the sequential order of conserved domains in a protein sequence.  see: https://www.ncbi.nlm.nih.gov/Structure/sparcle/docs/sparcle_about.html
Primarily, names and functional annotations are added through manual curation. However, it would be preferable if automated processes could support and supplement curation efforts. Currently, there are over 42k curated architechtures out of over 200k in the SPARCLE database.  Autamated processes to name could be based on using previously curated architectures to name related architectures. (because the CDD imports domain models from multiple resources, there can be numerous CDs that model overlapping protein footprints (clusters of these models are called 'superfmailies'). Therefore, numerous architectures could be given the same name ) Another approach could be to use name and labels attached to CDs and superfamilies within an arhictecture to predict a useful name for the architecture. 
![image](https://github.com/NCBI-Codeathons/mlxai-2024-team-gwadz-yang/assets/35601022/debf6388-734a-472d-9bdd-8a8013a4400f)
![image](https://github.com/NCBI-Codeathons/mlxai-2024-team-gwadz-yang/assets/35601022/72302944-3ad2-482f-a0d7-5ebd7a87f633)
![image](https://github.com/NCBI-Codeathons/mlxai-2024-team-gwadz-yang/assets/35601022/fc5f012e-0666-443a-a73c-212da92ad77f)



## Approach

#### Goal: 
Given a set of curated domain architectures, can we predict a good name for related architectures. 

Inputs: 
`a specific architecture string and a superfamily string`

Output:  `a  name for the Architecture (CurNamne in our data Table)`

e.g. Arch 1 has ArchString(specific CDs): `cd12718 cd16457 pfam02148`
and this superfamily string (the cluster that each of those CDs belongs to) : `zf-UBP RRM_SF RING_Ubox`
and a curated name: `BRCA1-associated protein`

can we predict a name for a related architecture with archstring: `cd12718 cd16457 pfam02148 cl34174` and superfamily string: 
`zf-UBP RRM_SF RING_Ubox Smc`  (curators gave this the same name) 


Categorization Model to auto-name SPARCLE architectures. (ordered arrangement of protein domains)

1. organize data in one-hot encoded matrix. 
	rows- curated architecture examples.  target value is curated names (28,055 categories, prior to simplification)
	features/columns all possible specific-hits (domain) or superfamilies (domain clusters)= 41,888 
2. simplify categories to eliminate overly specific names and to increase # examples/category 
3. choose >=1  ML models to evaluate data (e.g. Decisions trees, Deep Learning, Clustering?) 
4. choose strategy for training /test given sparse data
5. possible to add tokenized "title names" as features?
6. evaluate most promising model- tweak, if possible. 
7. option - test trained model vs uncurated data to estimate impact. 


## Results

## Future Work

## NCBI Codeathon Disclaimer
This software was created as part of an NCBI codeathon, a hackathon-style event focused on rapid innovation. While we encourage you to explore and adapt this code, please be aware that NCBI does not provide ongoing support for it.

For general questions about NCBI software and tools, please visit: [NCBI Contact Page](https://www.ncbi.nlm.nih.gov/home/about/contact/)

