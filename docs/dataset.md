# Dataset

## Canonical provenance

The canonical source is the Kaggle dataset page [`mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The data was collected by Worldline and the Machine Learning Group of Université Libre de Bruxelles. It describes European cardholders in September 2013 over two days: 284,807 transactions, including 492 frauds (0.172% of transactions).

The features are PCA-transformed `V1`–`V28`, plus untransformed `Time` and `Amount`. `Class` is the binary target.

## Schema

| Column | Description |
| --- | --- |
| `Time` | Seconds elapsed since the first transaction in the dataset. |
| `V1`–`V28` | PCA-transformed, anonymized numeric features. |
| `Amount` | Original transaction amount. |
| `Class` | Target: `0` = legitimate, `1` = fraud. |

## License and source discipline

The canonical Kaggle page states that the database is made available under the Open Database License (ODbL) 1.0 and its contents under the Database Contents License (DbCL) 1.0. **Verify the current terms on the canonical page before using or redistributing the dataset.** A third-party Zenodo re-upload (record [7395559](https://zenodo.org/records/7395559)) uses different terms, CC-BY-4.0, and is not the canonical source.

This repository does not redistribute the dataset. The user downloads the canonical data into `data/raw/`; raw data and trained artifacts are never committed. The repository ships synthetic fixtures only.

## Citation

A canonical associated paper is:

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi, “Calibrating Probability with Undersampling for Unbalanced Classification,” *IEEE Symposium on Computational Intelligence and Data Mining (CIDM)*, 2015.
