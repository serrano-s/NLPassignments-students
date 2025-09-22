"""
CSE 447/547- Project 2a
by kahuja
Adapted from from:
CS124 PA5: Quizlet // Stanford, Winter 2020
"""
import os
import json
import pandas as pd
import numpy as np
import torch
from glove import GloveEmbeddings
from sentence_transformers import SentenceTransformer


# Preload glove embeddings
parent_dir = os.path.dirname(os.path.abspath("__file__"))
data_dir = os.path.join(parent_dir, "data")
glove_embeddings = GloveEmbeddings(
    path=f"{data_dir}/embeddings/glove.6B/glove.6B.50d.txt"
)


def load_synonyms_data(split="dev", data_dir="data"):

    synonyms_df = pd.read_csv(f"{data_dir}/synonyms/{split}.csv", sep="\t")
    return synonyms_df


class Exercise1Runner():

    def __init__(self, find_synonym):
        self.find_synonym = find_synonym

        # load embeddings
        self.embeddings = glove_embeddings

        # load synonyms data
        self.synonyms_df = load_synonyms_data(split="dev", data_dir=data_dir)

    def evaluate(self, print_q=True):

        print("Excercise 1: Synonyms")
        print("-----------------")

        acc_cosine_sim = self.get_synonym_acc("cosine", print_q)
        acc_euclidean_dist = self.get_synonym_acc("euclidean", print_q)
        acc_manhattan_dist = self.get_synonym_acc("manhattan", print_q)

        print("accuracy using cosine similarity : %.5f" % acc_cosine_sim)
        print("accuracy using euclidean distance : %.5f" % acc_euclidean_dist)
        print("accuracy using manhattan distance : %.5f" % acc_manhattan_dist)

    def get_synonym_acc(self, metric, print_q=True):

        if print_q:
            print ('Answering exercise 1 using %s as the comparison metric...' % metric)

        num_correct = 0
        total = 0

        for i, row in self.synonyms_df.iterrows():
            word = row["word"]
            choices = row["choices"].split(",")
            correct_choice = row["true_answer"]

            synonym_dict = self.find_synonym(word, choices, self.embeddings, metric)
            synonym = synonym_dict["synonym"]

            if synonym == correct_choice:
                num_correct += 1
            total += 1

            if print_q:
                print("%d. What is a synonym for %s?" % (i + 1, word))
                a, b, c, d = choices[0], choices[1], choices[2], choices[3]
                print("    a) %s\n    b) %s\n    c) %s\n    d) %s" % (a, b, c, d))
                print("you answered: %s \n" % synonym)

        return num_correct / total


class Exercise2Runner():

    def __init__(self, find_analogy_word):

        self.find_analogy_word = find_analogy_word
        self.embeddings = glove_embeddings
        self.analogies_df = pd.read_csv(f"{data_dir}/analogies/dev.csv", sep="\t")

    def evaluate(self, print_q=True):
        print("Excercise 2: Analogies")
        print("-----------------")
        
        self.get_analogy_acc(print_q)
        pass

    def get_analogy_acc(self, print_q=True):
        if print_q:
            print ('Answering exercise 2...')

        num_correct = 0
        total = 0

        for i, row in self.analogies_df.iterrows():

            abs, choices = (
                row["a,b,aa,bb"].split(","),
                row["choices_list"].split(","),
            )
            a, b, aa, bb = abs
            correct_choice = bb

            analogy_answer = self.find_analogy_word(a, b, aa, choices, self.embeddings)

            if analogy_answer == correct_choice:
                num_correct += 1

            total += 1

            if print_q:
                print("%d. %s is to %s as %s is to ?" % (i + 1, a, b, aa))
                print("    a) %s\n    b) %s\n    c) %s\n    d) %s" % tuple(choices))
                print("You answered: %s \n" % analogy_answer)

        acc = num_correct / total
        print("accuracy using analogy questions : %.5f" % acc)
        print(" ")
        return acc


class Exercise3aRunner:

    def __init__(self, weat_effect_size):

        self.weat_effect_size = weat_effect_size
        self.embeddings = glove_embeddings

        with open(f"{data_dir}/weat/weat.json") as f:
            self.weat_data = json.load(f)

    def evaluate_effect_size(self, print_q=True):
        print("Excercise 3a: WEAT Effect Size")
        print("-----------------")

        flower_insect_pun_effect_size = self.get_weat_effect_size(
                "Flowers_Insects_Pleasant_Unpleasant",
                print_q,
            )

        music_weapons_pun_effect_size = self.get_weat_effect_size(
                "MusicalInstruments_Weapons_Pleasant_Unpleasant",
                print_q
            )
        eur_afr_pun_effect_size = self.get_weat_effect_size(
                        "EuropeanAmerican_AfricanAmerican_Pleasant_Unpleasant",
                        print_q
                    )
        male_female_carfam_effect_size = self.get_weat_effect_size("Male_Female_Career_Family", print_q)

        math_art_malefemale_effect_size = self.get_weat_effect_size(
            "Math_Arts_Male_Female", print_q
        )

        print(
            "Effect size for Flowers-Insects Pleasant Unpleasant case: %.5f"
            % flower_insect_pun_effect_size
        )
        print(
            "Effect size for MusicalInstruments-Weapons Pleasant Unpleasant case: %.5f"
            % music_weapons_pun_effect_size
        )
        print(
            "Effect size for EuropeanAmerican AfricanAmerican_Pleasant_Unpleasant case: %.5f"
            % eur_afr_pun_effect_size
        )
        print(
            "Effect size for Male_Female Career_Family case : %.5f"
            % male_female_carfam_effect_size
        )

        print(
            "Effect size for Math_Art Male_Female case : %.5f" % math_art_malefemale_effect_size
        )

    def get_weat_effect_size(self, weat_dataset_type, print_q=True):

        if print_q:
            print ('Answering exercise 3 using %s as the dataset...' % weat_dataset_type)

        weat_data = self.weat_data[weat_dataset_type]
        X, Y, A, B = weat_data[weat_data["X_key"]], weat_data[weat_data["Y_key"]], weat_data[weat_data["A_key"]], weat_data[weat_data["B_key"]]

        if print_q:
            print("Target Group X: %s" % X)
            print("Target Group Y: %s" % Y)
            print("Attribute Group A: %s" % A)
            print("Attribute Group B: %s" % B)
            print("")

        effect_size = self.weat_effect_size(X, Y, A, B, self.embeddings)
        return effect_size


class Exercise3bRunner:

    def __init__(self, weat_p_value):

        self.weat_p_value = weat_p_value
        self.embeddings = glove_embeddings

        with open(f"{data_dir}/weat/weat.json") as f:
            self.weat_data = json.load(f)

    def evaluate(self, print_q=True):
        print("Excercise 3b: WEAT P-Value")
        print("-----------------")

        eur_afr_pun_p_value = self.get_weat_p_value(
            "EuropeanAmerican_AfricanAmerican_Pleasant_Unpleasant", print_q
        )

        male_female_carfam_p_value = self.get_weat_p_value("Male_Female_Career_Family", print_q)

        math_art_malefemale_p_value = self.get_weat_p_value("Math_Arts_Male_Female", print_q)

        print("P-value for EuropeanAmerican AfricanAmerican_Pleasant_Unpleasant case: %.5f" % eur_afr_pun_p_value)
        print(
            "P-value for Male Female Career Family case : %.5f"
            % male_female_carfam_p_value
        )
        print("P-value for Math_Art Male_Female case %.5f" % math_art_malefemale_p_value)

    def get_weat_p_value(self, weat_dataset_type, print_q=True):

        if print_q:
            print ('Answering exercise 3 using %s as the dataset...' % weat_dataset_type)

        weat_data = self.weat_data[weat_dataset_type]
        X, Y, A, B = weat_data[weat_data["X_key"]], weat_data[weat_data["Y_key"]], weat_data[weat_data["A_key"]], weat_data[weat_data["B_key"]]

        if print_q:
            print("Target Group X: %s" % X)
            print("Target Group Y: %s" % Y)
            print("Attribute Group A: %s" % A)
            print("Attribute Group B: %s" % B)
            print("")

        p_value = self.weat_p_value(X, Y, A, B, self.embeddings, max_permutations=1000)
        return p_value

class Exercise4Runner:

    def __init__(self, get_sentence_similarity):
        self.get_sentence_similarity = get_sentence_similarity
        self.embeddings = glove_embeddings
        self.sentences_df = pd.read_csv(
            f"{data_dir}/sentence_similarity/dev.csv",
            sep="\t",
            header=None,
            names=["label", "sentence1", "sentence2"],
        )

        with open(f"{data_dir}/pos_weights.txt", "r") as f:
            pos_weights = f.read().split("\n")
            pos_weights = {line.split()[0]: float(line.split()[1]) for line in pos_weights}
        self.pos_weights = pos_weights

    def evaluate(self, print_q=True):
        print("Excercise 4: Sentence Similarity")
        print("-----------------")

        acc_sum = self.get_sentence_similarity_acc(print_q, use_POS=False)
        acc_pos_sum = self.get_sentence_similarity_acc(print_q, use_POS=True)

        print("accuracy using sum of word vectors : %.5f" % acc_sum)
        print("accuracy using sum of word vectors with POS weights : %.5f" % acc_pos_sum)

    def get_sentence_similarity_acc(self, print_q=True, use_POS=False):
        if print_q:
            print ('Answering exercise 4...')

        THRESHOLD = 0.95
        num_correct = 0
        total = 0

        for i, row in self.sentences_df.iterrows():
            sentence1, sentence2 = row["sentence1"], row["sentence2"]
            label = row["label"]

            similarity_cos = self.get_sentence_similarity(sentence1, sentence2, self.embeddings, use_POS=use_POS, pos_weights=self.pos_weights)
            similarity_pred = 1 if similarity_cos > THRESHOLD else 0
            if similarity_pred == label:
                num_correct += 1
            total += 1

            if print_q:
                print("%d. Are the following sentences similar?" % (i + 1))
                print("    Sentence 1: %s" % sentence1)
                print("    Sentence 2: %s" % sentence2)
                print(" You answered: Cosine Similarity = %.5f, Prediction = %d" % (similarity_cos, similarity_pred))
                print(" ")
        acc = num_correct / total
        # print("accuracy using sentence similarity questions : %.5f" % acc)
        # print(" ")
        return acc


class Exercise5Runner:

    def __init__(self, GloveKNNClassifier):
        self.cls_class = GloveKNNClassifier
        self.embeddings = glove_embeddings

        self.train_df = pd.read_csv(
            f"{data_dir}/sst/sst_train_binary.csv"
        )
        self.dev_df = pd.read_csv(f"{data_dir}/sst/sst_dev_binary.csv")

        self.train_df_multiclass = pd.read_csv(
            f"{data_dir}/sst/sst_train_multiclass.csv"
        )
        self.dev_df_multiclass = pd.read_csv(f"{data_dir}/sst/sst_dev_multiclass.csv")

        with open(f"{data_dir}/pos_weights.txt", "r") as f:
            pos_weights = f.read().split("\n")
            pos_weights = {line.split()[0]: float(line.split()[1]) for line in pos_weights}
        self.pos_weights = pos_weights

    def evaluate(self, k=5, return_vals = False):

        print("Excercise 5: KNN Classifier")
        print("-----------------")

        print(f"Binary Classification using k = {k} using sum of word vectors")
        train_acc_bc_sum = self.get_knn_acc(
            self.train_df["sentence"].values.tolist(),
            self.train_df["label"].values.tolist(),
            self.train_df["sentence"].values.tolist(),
            self.train_df["label"].values.tolist(),
            k=k,
        )
        dev_acc_bc_sum = self.get_knn_acc(
            self.train_df["sentence"].values.tolist(),
            self.train_df["label"].values.tolist(),
            self.dev_df["sentence"].values.tolist(),
            self.dev_df["label"].values.tolist(),
            k=k,
        )
        print("Accuracy on train set: %.5f" % train_acc_bc_sum)
        print("Accuracy on dev set: %.5f" % dev_acc_bc_sum)
        print(" ")

        print(f"Multiclass Classification using k = {k} using sum of word vectors")
        train_acc_mc_sum = self.get_knn_acc(
            self.train_df_multiclass["sentence"].values.tolist(),
            self.train_df_multiclass["label"].values.tolist(),
            self.train_df_multiclass["sentence"].values.tolist(),
            self.train_df_multiclass["label"].values.tolist(),
            k=k,
        )
        dev_acc_mc_sum = self.get_knn_acc(
            self.train_df_multiclass["sentence"].values.tolist(),
            self.train_df_multiclass["label"].values.tolist(),
            self.dev_df_multiclass["sentence"].values.tolist(),
            self.dev_df_multiclass["label"].values.tolist(),
            k=k,
        )
        print("Accuracy on train set: %.5f" % train_acc_mc_sum)
        print("Accuracy on dev set: %.5f" % dev_acc_mc_sum)
        print(" ")

        print(f"Binary Classification using k = {k} using sum of word vectors with POS weights")
        train_acc_bc_psum = self.get_knn_acc(
            self.train_df["sentence"].values.tolist(),
            self.train_df["label"].values.tolist(),
            self.train_df["sentence"].values.tolist(),
            self.train_df["label"].values.tolist(),
            k=k,
            use_POS=True
        )
        dev_acc_bc_psum = self.get_knn_acc(
            self.train_df["sentence"].values.tolist(),
            self.train_df["label"].values.tolist(),
            self.dev_df["sentence"].values.tolist(),
            self.dev_df["label"].values.tolist(),
            k=k,
            use_POS=True
        )
        print("Accuracy on train set: %.5f" % train_acc_bc_psum)
        print("Accuracy on dev set: %.5f" % dev_acc_bc_psum)
        print(" ")

        print(f"Multiclass Classification using k = {k} using sum of word vectors with POS weights")
        train_acc_mc_psum = self.get_knn_acc(
            self.train_df_multiclass["sentence"].values.tolist(),
            self.train_df_multiclass["label"].values.tolist(),
            self.train_df_multiclass["sentence"].values.tolist(),
            self.train_df_multiclass["label"].values.tolist(),
            k=k,
            use_POS=True
        )
        dev_acc_mc_msum = self.get_knn_acc(
            self.train_df_multiclass["sentence"].values.tolist(),
            self.train_df_multiclass["label"].values.tolist(),
            self.dev_df_multiclass["sentence"].values.tolist(),
            self.dev_df_multiclass["label"].values.tolist(),
            k=k,
            use_POS=True
        )
        print("Accuracy on train set: %.5f" % train_acc_mc_psum)
        print("Accuracy on dev set: %.5f" % dev_acc_mc_msum)
        
        if return_vals:
            return {
                "binary": {
                    "sum": {
                        "train": train_acc_bc_sum,
                        "dev": dev_acc_bc_sum
                    },
                    "pos_weight_sum": {
                        "train": train_acc_bc_psum,
                        "dev": dev_acc_bc_psum
                    }
                },
                
                "multiclass": {
                    "sum": {
                        "train": train_acc_mc_sum,
                        "dev": dev_acc_mc_sum
                    },
                    "pos_weight_sum": {
                        "train": train_acc_mc_psum,
                        "dev": dev_acc_mc_msum
                    }
                }
            }
    def get_knn_acc(self, X_train, y_train, X_dev, y_dev, k = 5, use_POS=False):

        knn_cls = self.cls_class(
            self.embeddings,
            k=k,
            use_POS=use_POS,
            pos_weights=self.pos_weights
        )

        knn_cls.fit(X_train, y_train)

        y_pred = knn_cls.predict(X_dev)

        y_dev_np = np.array(y_dev)
        y_pred_np = np.array(y_pred)
        acc = (y_dev_np == y_pred_np).mean()

        return acc


class Exercise6Runner:

    def __init__(self, SentenceTransformerKNNClassifier):

        self.cls_class = SentenceTransformerKNNClassifier

        self.train_df = pd.read_csv(f"{data_dir}/sst/sst_train_binary.csv")
        self.dev_df = pd.read_csv(f"{data_dir}/sst/sst_dev_binary.csv")

        self.train_df_multiclass = pd.read_csv(
            f"{data_dir}/sst/sst_train_multiclass.csv"
        )
        self.dev_df_multiclass = pd.read_csv(f"{data_dir}/sst/sst_dev_multiclass.csv")

    def evaluate(self, k=5, debug=False, st_model="all-mpnet-base-v2", return_vals=False):

        print("Excercise 6: Sentence Transformer KNN Classifier")
        print("-----------------")

        if debug:
            print("Running on Debug Model. Will only use 100 examples in training and 10 examples for dev set")
            print(" ")
            train_sents = self.train_df["sentence"].values.tolist()[:100]
            train_labels = self.train_df["label"].values.tolist()[:100]
            train_labels_multi = self.train_df_multiclass["label"].values.tolist()[:100]
            dev_sents = self.dev_df["sentence"].values.tolist()[:10]
            dev_labels = self.dev_df["label"].values.tolist()[:10]
            dev_labels_multi = self.dev_df_multiclass["label"].values.tolist()[:10]

        else:
            train_sents = self.train_df["sentence"].values.tolist()
            train_labels = self.train_df["label"].values.tolist()
            train_labels_multi = self.train_df_multiclass["label"].values.tolist()
            dev_sents = self.dev_df["sentence"].values.tolist()
            dev_labels = self.dev_df["label"].values.tolist()
            dev_labels_multi = self.dev_df_multiclass["label"].values.tolist()
        
        print(f"Binary Classification using k = {k} using Sentence Transformer")
        dev_acc_bc_st = self.get_knn_acc(
            train_sents,
            train_labels,
            dev_sents,
            dev_labels,
            st_model=st_model,
            k=k,
        )
        print("Accuracy on dev set: %.5f" % dev_acc_bc_st)

        print(" ")

        print(f"Multiclass Classification using k = {k} using Sentence Transformer")

        dev_acc_bc_st = self.get_knn_acc(
            train_sents,
            train_labels_multi,
            dev_sents,
            dev_labels_multi,
            st_model=st_model,
            k=k,
        )

        print("Accuracy on dev set: %.5f" % dev_acc_bc_st)

        if return_vals:
            return {
                "binary": {
                    "dev": dev_acc_bc_st
                },
                "multiclass": {
                    "dev": dev_acc_bc_st
                }
            }

    def get_knn_acc(
        self, X_train, y_train, X_dev, y_dev, st_model="all-mpnet-base-v2", k=5
    ):

        knn_cls = self.cls_class(st_model, k=k)

        knn_cls.fit(X_train, y_train)

        y_pred = knn_cls.predict(X_dev)

        y_dev_np = np.array(y_dev)
        y_pred_np = np.array(y_pred)
        acc = (y_dev_np == y_pred_np).mean()

        return acc
