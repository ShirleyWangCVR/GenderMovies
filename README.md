# CSC2552 Project: Improvements in Gender Representation in Movies Over the Past Century

Author: Shirley Wang

## Data

This exploration uses three datasets:
1. [Kaggle Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
2. [Bechdel Test](https://bechdeltest.com/) (data can be accessed through making a request to their API)
3. [MovieGalaxies](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/T4HBA3)

To use, please download these three datasets and place them in a folder `data`

```bash
├── data
│   ├── MovieGalaxies
│   │   ├── gexf
│   │   ├── network_metadata.tab
│   ├── MoviesDataset
│   ├── bechdeltest.json
```

## Environment

Programming was done with Python 3.10. Please ensure the following packages are installed in your environment:

```bash
matplotlib
numpy
pandas
seaborn
sklearn
statsmodels
torch
torch-geometric
transformers
```

## Code

All programming and visualizations can be found in `TrainingAndAnalysis.ipynb`. Rough work and more exploration into using graphs and self-supervised learning for MovieGalaxies can be found in `RoughWork.ipynb`, although through experimentation, it does not seem like an effective method. 

Code in `GraphSSL` is used for graph self-supervised learning, although it is ineffective. The code is borrowed and modified from [this medium article](https://medium.com/stanford-cs224w/self-supervised-learning-for-graphs-963e03b9f809).

`dataset.py` and `model.py` are used for the gender labelling portion of the report.
