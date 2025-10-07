The embeddings in `lyrics-embeddings.npy` *used* to be:

- a subset from Stanford's pretrained 50-dimensional Wikipedia + Gigaword 5 GloVe embeddings (https://nlp.stanford.edu/projects/glove/) to correspond to all words in `lyrics-train.csv`.

But now, they're instead 20-dimensional embeddings pretrained exclusively on `lyrics-train.csv` using gensim.
