# PACL
We proposed a novel pathway-based deep clustering method (PACL) for molecular subtyping of cancer by incorporating gene expression and biological pathway database.<br/>
PACL main contribution is to capture nonlinear associations between high-dimensional gene expression data and patients’ survivals and to interpret the model biologically.

To evaluate performance of PACL, we conducted experiments with highdimensional gene expression data of patients in Glioblastoma
Multiforme (GBM) and ovarian cancer./
Downloaded the datasets of [data_expression.txt](https://github.com/cBioPortal/datahub/blob/master/public/gbm_tcga) and [data_bcr_clinical_data_patient.txt](https://github.com/cBioPortal/datahub/blob/master/public/gbm_tcga) of GBM cancer.
Then gene expression data is preprocessed such that all the samples in gene expression data has respective survival months and status from 
clinical data.
# Requirements
[Python-3.5.2](https://www.python.org/downloads/release/python-352/)<br/>
[numpy-1.14.2](http://www.numpy.org/)<br/>
[pandas-0.22.0](https://pandas.pydata.org/pandas-docs/version/0.22/whatsnew.html)<br/>
[torch-0.4.0](https://pytorch.org/get-started/previous-versions/) <br/>
[scipy-1.2.1](https://pypi.org/project/scipy/) <br/>
