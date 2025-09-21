import torch

class GloveEmbeddings:

    def __init__(self, path="embeddings/glove.6B/glove.6B.50d.txt"):
        self.path = path
        self.embeddings = {}
        self.load()

    def load(self):

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines:
            # print(line)
            # print(line.split())
            values = line.split()
            word = values[0]
            # print(list(map(float, values[1:])))
            vector = torch.tensor(list(map(float, values[1:])), dtype=torch.float)
            self.embeddings[word] = vector

    def is_word_in_embeddings(self, word):
        return word in self.embeddings

    def get_vector(self, word):
        if not self.is_word_in_embeddings(word):
            return self.embeddings["unk"]
        return self.embeddings[word]

    # Use square operator to get the vector of a word
    def __getitem__(self, word):
        return self.get_vector(word)
