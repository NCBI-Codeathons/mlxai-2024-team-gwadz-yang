# SPARCLE Curation
## Team Gwadz-Yang

In order to assist a small team of experts manually annotating subfamily protein architectures in the SPARCLE database, we trained a decision tree and an NLP model on the roughly 42k expert-curated architectures in order to automate assignment of labels to the remaining roughly 200k uncurated protein architectures. 


## Table of Contents

1. [Introduction](#introduction)
2. [Methods](#methods)
3. [Results](#results)
4. [Discussion & Conclusion](#discussion--conclusion)
5. [References](#references)
6. [Acknowledgement](#acknowledgements)
7. [Team Members](#team-members)

## INTRODUCTION

SPARCLE, the Subfamily Protein Architecture Labeling Engine, is a resource for the functional characterization and labeling of protein sequences that have been grouped by their characteristic conserved domain architecture. 


A domain architecture is defined as the sequential order of conserved domains in a protein sequence.  see: https://www.ncbi.nlm.nih.gov/Structure/sparcle/docs/sparcle_about.html


Primarily, names and functional annotations are added through manual curation. However, it would be preferable if automated processes could support and supplement curation efforts. Currently, there are over 42k curated architechtures out of over 200k in the SPARCLE database.  Autamated processes to name could be based on using previously curated architectures to name related architectures. (because the CDD imports domain models from multiple resources, there can be numerous CDs that model overlapping protein footprints (clusters of these models are called 'superfmailies'). Therefore, numerous architectures could be given the same name ) Another approach could be to use name and labels attached to CDs and superfamilies within an arhictecture to predict a useful name for the architecture. 


![image](https://github.com/NCBI-Codeathons/mlxai-2024-team-gwadz-yang/assets/35601022/debf6388-734a-472d-9bdd-8a8013a4400f)
![image](https://github.com/NCBI-Codeathons/mlxai-2024-team-gwadz-yang/assets/35601022/72302944-3ad2-482f-a0d7-5ebd7a87f633)
![image](https://github.com/NCBI-Codeathons/mlxai-2024-team-gwadz-yang/assets/35601022/fc5f012e-0666-443a-a73c-212da92ad77f)

![image](https://github.com/NCBI-Codeathons/mlxai-2024-team-gwadz-yang/assets/35601022/ac748a8c-7974-4c79-9f4b-58efddaacf88)


### Scope and Research Questions

* RQ1: Given a set of curated domain architecture names (`CurName`), can we predict a good name for related architectures based on the specific domain architectures (`SpecificArch`) and the superfamily architecures (`superfamilyarch`).

* RQ2: Will addition of architecture title strings (`TitleString`) to the input matrix improve prediction accuracy of curated names (`CurName`)?

![image](docs/images/FA_slide_training_testing.png)


## METHODS

### Overall Strategy


Categorization Model to auto-name SPARCLE architectures (ordered arrangement of protein domains).

1. Preprocess the output variables by simplifying the ~14k original curated names to 11.8k to eliminate overly specific names and increase the number of examples per category.
2. Preprocess the input variables by lowercasing and removal of uninformative, common words, like prepositions, articles, and non-specific terms like "domain" and "protein".
3. Organize curated input/output data variables into encoded matrix using SentencePiece tokenization.
4. Train a decision tree model using the SentencePiece encoded matrix.
5. Add tokenized "title strings" as features to evaluate whether this sparsely populated, additional data improves model accuracy.
6. Test trained model vs. uncurated data to estimate impact.

### RQ1 Methods

* Inputs: 
A matrix of SentencePiece tokenized curated names (`CurName`) and associated specific (`SpecificArch`) and superfamily (`superfamilyarch`) architectures derived from the SPARCLE table of 42,766 curarted protein architectures.


* Output: 
Curated names (`CurName`) for uncurated architectures given their specific conserved domains (`SpecificArch`) and superfaimly architectures (`superfamilyarch`).


* Example: 
Architecture 1 (`ArchID` 1) has specific CDs (`SpecificArch`): `cd12718 cd16457 pfam02148`, and this superfamily string (`superfamilyArch`, the cluster that each of those CDs belongs to): `zf-UBP RRM_SF RING_Ubox` and a curated name: `BRCA1-associated protein`. Can we predict a name for a related architecture with archstring: `cd12718 cd16457 pfam02148 cl34174` and superfamily string: `zf-UBP RRM_SF RING_Ubox Smc`? Expert curators have given this the same name.


### RQ2 Methods

* The same methodology will be used for RQ2 with the addition of SentencePiece tokenized `TitleStrings` to the input vectors.


## Results

In order to evaluate the accuracy of curated name predictions for the protein domains, we 

## Discussion & Conclusion

Much of our time was dedicated to cleaning the original SPARCLE data and evaluating different input variable tokenization methods.

Given more time, we would have liked to train and evaluate multiple NLP models and experiment with various tokenization methods and the influence of including different input variables, like `TitleStrings`. However, given the time constraints of the codeathon, we were only able to get preliminary results for our decision tree model. We also lacked time to dedicate to exploring model validation strategies. 


## References
Marchler-Bauer A, Bo Y, Han L, He J, Lanczycki CJ, Lu S, Chitsaz F, Derbyshire MK, Geer RC, Gonzales NR, Gwadz M, Hurwitz DI, Lu F, Marchler GH, Song JS, Thanki N, Wang Z, Yamashita RA, Zhang D, Zheng C, Geer LY, Bryant SH. CDD/SPARCLE: functional classification of proteins via subfamily domain architectures. Nucleic Acids Res. 2017 Jan 4;45(D1):D200-D203. doi: 10.1093/nar/gkw1129. Epub 2016 Nov 29. [PubMed PMID: 27899674] [Full Text at Oxford Academic]

Kudo T, Richardson J. (2018) SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. arXiv:1808.06226v1 [cs.CL] https://doi.org/10.48550/arXiv.1808.06226


## Future Work

## NCBI Codeathon Disclaimer
This software was created as part of an NCBI codeathon, a hackathon-style event focused on rapid innovation. While we encourage you to explore and adapt this code, please be aware that NCBI does not provide ongoing support for it.

For general questions about NCBI software and tools, please visit: [NCBI Contact Page](https://www.ncbi.nlm.nih.gov/home/about/contact/)

## ACKNOWLEDGEMENTS



## TEAM MEMBERS

* Marc Gwadz, NCBI, NLM, NIH (Team Leader)
* Mingzhang Yang, NCBI, NLM, NIH (Technical Lead)
* Christopher Meyer, University of Chicago (Writer)
* Franziska Ahrend, ORISE Fellow at NIDDK, NIH
* Yixiang Deng, Post-doc  MIT, Harvard
* Shaojun Xie, Advanced Biomedical and Computational Science, Frederick National Laboratory for Cancer Research, Frederick, MD, USA
