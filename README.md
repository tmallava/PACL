# PACL
We proposed a novel pathway-based deep clustering method (PACL) for molecular subtyping of cancer by incorporating gene expression and biological pathway database.<br/>
PACL main contribution is to capture nonlinear associations between high-dimensional gene expression data and patients’ survivals and to interpret the model biologically.

To evaluate performance of PACL, we conducted experiments with highdimensional gene expression data of patients in Glioblastoma
Multiforme (GBM) and ovarian cancer.<br/>
# Datasets:
## GBM Cancer
[gbm_gene_expression.txt](https://github.com/cBioPortal/datahub/blob/master/public/gbm_tcga/data_expression.txt) <br/>
[gbm_clinical_data.txt](https://github.com/cBioPortal/datahub/blob/master/public/gbm_tcga/data_bcr_clinical_data_patient.txt)<br/>

## Ovarian Cancer
[Ovarian_gene_expression.txt](https://github.com/cBioPortal/datahub/blob/master/public/ov_tcga/data_expression.txt)<br/> [Ovarian_clinical_data.txt](https://github.com/cBioPortal/datahub/blob/master/public/ov_tcga/data_bcr_clinical_data_patient.txt)<br/>

## Pathway data
[Pathway_databases](https://github.com/tmallava/PACL/blob/master/pathway(Gene).txt)<br/>

# Preprocessing
After downloading the datasets, gbm gene expression data is preprocessed such that all the samples in gene expression data has respective survival months and survival status from gbm clinical data.<br/>
From pathway data, we considered only four pathway databases: KEGG, Reactome, PID, and BioCarta for pathway-based analysis. <br/>
Small pathways which include less than 15 genes were excluded to avoid substantial redundancy with large pathways, and also
genes that have no association with pathways were not considered for the experiments.<br/>
Similar procedure is followed for Ovarian cancer too.<br/>
After the preprocessing, each cancer dataset has 998 pathways of 6,073 genes.<br/>
The experiments were repeated ten times with randomly selecting 80% samples for reproducibility and robustness.<br/>
For each experiment, data was normalized to mean of zero and a standard deviation of one.<br/>

# Results
Silhouette score is used to determine the optimal number of clusters (i.e., the number of subtypes) and Kaplan-Meier survival analysis to identify the subtypes are clinically distinct. 





# Requirements
[Python-3.5.2](https://www.python.org/downloads/release/python-352/)<br/>
[numpy-1.14.2](http://www.numpy.org/)<br/>
[pandas-0.22.0](https://pandas.pydata.org/pandas-docs/version/0.22/whatsnew.html)<br/>
[torch-0.4.0](https://pytorch.org/get-started/previous-versions/) <br/>
[scipy-1.2.1](https://pypi.org/project/scipy/) <br/>
