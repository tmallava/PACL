# PACL
We proposed a novel pathway-based deep clustering method (PACL) for molecular subtyping of cancer by incorporating gene expression and biological pathway database.<br/>
PACL main contribution is to capture nonlinear associations between high-dimensional gene expression data and patients’ survivals and to interpret the model biologically.

To evaluate performance of PACL, we conducted experiments with highdimensional gene expression data of patients in Glioblastoma
Multiforme (GBM) and ovarian cancer.<br/>
# Datasets:
## GBM cancer
[gbm_gene_expression.txt](https://github.com/cBioPortal/datahub/blob/master/public/gbm_tcga/data_expression.txt) 
[gbm_clinical_data.txt](https://github.com/cBioPortal/datahub/blob/master/public/gbm_tcga/data_bcr_clinical_data_patient.txt)

## Ovarian Cancer
[Ovarian_gene_expression.txt](https://github.com/cBioPortal/datahub/blob/master/public/ov_tcga/data_expression.txt) [Ovarian_clinical_data.txt](https://github.com/cBioPortal/datahub/blob/master/public/ov_tcga/data_bcr_clinical_data_patient.txt)
We downloaded the GBM cancer datasets of [gbm_gene_expression.txt](https://github.com/cBioPortal/datahub/blob/master/public/gbm_tcga/data_expression.txt) & [gbm_clinical_data.txt](https://github.com/cBioPortal/datahub/blob/master/public/gbm_tcga/data_bcr_clinical_data_patient.txt) and Ovarian cancer datasets [Ovarian_gene_expression.txt](https://github.com/cBioPortal/datahub/blob/master/public/ov_tcga/data_expression.txt)& [Ovarian_clinical_data.txt](https://github.com/cBioPortal/datahub/blob/master/public/ov_tcga/data_bcr_clinical_data_patient.txt).<br/>
Then gene expression data is preprocessed such that all the samples in gene expression data has respective survival months and survival status from clinical data.
We considered the four pathway databases: Kyoto
Encyclopedia of Genes and Genomes (KEGG), Reactome,
Pathway Interaction Database (PID), and BioCarta for
pathway-based analysis. The pathway databases were obtained
from Molecular Signatures Database (MSigDB)



# Requirements
[Python-3.5.2](https://www.python.org/downloads/release/python-352/)<br/>
[numpy-1.14.2](http://www.numpy.org/)<br/>
[pandas-0.22.0](https://pandas.pydata.org/pandas-docs/version/0.22/whatsnew.html)<br/>
[torch-0.4.0](https://pytorch.org/get-started/previous-versions/) <br/>
[scipy-1.2.1](https://pypi.org/project/scipy/) <br/>
