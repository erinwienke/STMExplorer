# STMExplorer: multi-view graph contrastive learning of spatial transcriptomics for tumor microenvironment discovery in colorectal cancer

STMExplorer is an automated, reproducible framework for tumor microenvironment (TME) discovery in colorectal cancer (CRC) using graph contrastive learning on spatial transcriptomics data.  At each training epoch, feature masking is used to generate two augmented views of the same graph. Both views are encoded by a shared two-layer GCN and optimized with a symmetric contrastive loss to learn low-dimensional node embeddings. Final embeddings are then obtained by encoding the full, unmasked feature matrix and are subsequently clustered using K-means and evaluated with standard clustering metrics.

![Figure](https://github.com/erinwienke/STMExplorer/blob/main/STMExplorer.png?raw=true)

## Requirements
* Python (>= 3.10)
* scikit-learn (>= 1.2)
* Python packages: anndata, scanpy, numpy, pandas, scipy, torch, torch_geometric

## Usage
Clone the repository or download source code files.

## Data 
The annotated spatial transcriptomics dataset utilized in this study is available through Zenodo at [https://zenodo.org/records/7760264](https://zenodo.org/records/7760264). The two unannotated datasets are available via the Gene Expression Omnibus (GEO) database under the accession codes of [GSE280313](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280313) and [GSE226997](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE226997). Due to the size of the data, only one sample is included in the repository for experiment replication (SN124_A595688_Rep2 from the annotated dataset).

## Preprocessing 

## Identifying TME regions from an annotated sample

## Identifying TME regions from an unannotated sample

## Performing an ablation experiment

## Contact
If you have any questions or problems, please send an email to erinw01 AT vt.edu

