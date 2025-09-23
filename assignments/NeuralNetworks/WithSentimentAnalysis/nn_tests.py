"""
CSE 447/547- Project 2b
by kahuja
"""
import os
from typing import Dict
import pandas as pd
import numpy as np
import torch
from siqa import load_siqa_data
from sentence_transformers import SentenceTransformer

parent_dir = os.path.dirname(os.path.abspath("__file__"))
data_dir = os.path.join(parent_dir, "data")

class Exercise1Runner:

    def __init__(self, FFNN):

        self.nn_class = FFNN

    def evaluate(self):
        print("Running Exercise 1: Implementing Feed-Forward Neural Networks in Pytorch")
        print("-----------------")
        
        print("Test Case 1: Testing if output shape is correct for binary classifier case")
        ffnn = self.nn_class(
            input_dim=5,
            hidden_dim=10,
            num_classes=1,
        )
        inp_x = torch.zeros(4, 5)
        out = ffnn(inp_x)
        print("Input Tensor: ", inp_x)
        print("Output shape: ", out.shape)
        print("Expected Output shape: torch.Size([4, 1])")
        
        print("\nTest Case 2: Testing if output shape is correct for multi-class classifier case")
        ffnn = self.nn_class(
            input_dim=5,
            hidden_dim=10,
            num_classes=3,
        )
        inp_x = torch.zeros(4, 5)
        out = ffnn(inp_x)
        print("Input Tensor: ", inp_x)
        print("Output shape: ", out.shape)
        print("Expected Output shape: torch.Size([4, 3])")
        
        print("\nTest Case 3: Testing if there are correct number of parameters in the model")
        ffnn = self.nn_class(
            input_dim=5,
            hidden_dim=10,
            num_classes=3,
        )
        num_params = sum(p.numel() for p in ffnn.parameters())
        print(f"Model: {ffnn}")
        print("Number of parameters: ", num_params)
        print("Expected Number of parameters: 93")
        
        
        print("\nTest Case 4: Testing if forward pass is correct gives expected results")
        torch.manual_seed(42)
        ffnn = self.nn_class(
            input_dim=5,
            hidden_dim=10,
            num_classes=3,
        )
        inp_x = torch.zeros(4, 5)
        out = ffnn(inp_x)
        print("Input Tensor: ", inp_x)
        print("Output Tensor: ", out)
        print(f"Expected Output Tensor: ", torch.zeros(4, 3))

class Exercise2Runner:

    def __init__(self, train, FFNN, datasets: Dict[str, torch.Tensor]):
        self.train = train
        self.nn_class = FFNN
        self.datasets = datasets

    def evaluate(self, embeddings="glove", multiclass=False, device="cpu"):
        assert embeddings in ["glove", "st"]

        main_key = "multiclass" if multiclass else "binary"

        X_train, y_train = self.datasets[main_key][embeddings]["X_train"], self.datasets[main_key][embeddings]["y_train"]
        X_dev, y_dev = self.datasets[main_key][embeddings]["X_dev"], self.datasets[main_key][embeddings]["y_dev"]

        print("Running Exercise 2: Training Feed-Forward Neural Networks in Pytorch")
        print("-----------------")

        print("\n Running using {} embeddings and {} classification".format(embeddings, "multiclass" if multiclass else "binary"))
        torch.manual_seed(42)
        nn_model = self.nn_class(
            input_dim=X_train.shape[1],
            hidden_dim=1024,
            num_classes=5 if multiclass else 1,
        )
        print("Model: ", nn_model)

        print("\nTraining the model")
        print("Using the following hyperparameters:")
        print("Learning Rate: 5e-5")
        print("Batch Size: 32")
        print("Number of Epochs: 20" if embeddings == "glove" else 10)
        print("Training Starting...")
        train_loss, dev_metrics = self.train(
            nn_model,
            X_train_embed=X_train,
            y_train=y_train,
            X_dev_embed=X_dev,
            y_dev=y_dev,
            lr=5e-5,
            batch_size=32,
            n_epochs=20 if embeddings == "glove" else 10,
            device=device,
            verbose=True
        )
        
        print("\nTraining Completed")
        print("\nTraining Loss: ", train_loss[-1])
        print("Dev Metrics: ", dev_metrics[-1])
        
        return nn_model, train_loss, dev_metrics


class Exercise3Runner:

    def __init__(self, predict, models, glove_embeddings, st_model):

        self.predict = predict
        self.models = models
        self.glove_embeddings = glove_embeddings
        self.st_model = st_model

        self.dev_df = pd.read_csv(f"{data_dir}/sst/sst_dev_binary.csv")
        self.dev_df_multiclass = pd.read_csv(f"{data_dir}/sst/sst_dev_multiclass.csv")

    def evaluate(self, multiclass=False, device="cpu", print_q=True):

        main_key = "multiclass" if multiclass else "binary"

        print("Running Exercise 3: Predicting with Feed-Forward Neural Networks in Pytorch")
        print("-----------------")

        print("\n Running {} classification".format("multiclass" if multiclass else "binary"))
        print("\nPredicting on the dev set using Glove FFNN model")

        glove_nn_preds = self.predict(
            sentences=self.dev_df["sentence"].values.tolist(),
            model=self.models[main_key]["glove"],
            embedding_method="glove",
            device=device,
            glove_embeddings=self.glove_embeddings,
        )

        print("\nPredicting on the dev set using ST FFNN model")
        st_nn_preds = self.predict(
            sentences=self.dev_df["sentence"].values.tolist(),
            model=self.models[main_key]["st"],
            embedding_method="st",
            device=device,
            st_model=self.st_model,
        )
        
        if print_q:
            print("\nPrinting Predictions for first 32 sentences")
            for sentence, glove_pred, st_pred in zip(self.dev_df["sentence"].values.tolist()[:32], glove_nn_preds, st_nn_preds):
                print("\nSentence: ", sentence)
                print("Glove Prediction: ", glove_pred)
                print("ST Prediction: ", st_pred)
                print("-------------------------------------------------")
        
        if multiclass:
            glove_acc = (np.array(glove_nn_preds) == self.dev_df_multiclass["label"].values).mean()
            st_acc = (np.array(st_nn_preds) == self.dev_df_multiclass["label"].values).mean()
        else:
            glove_acc = (np.array(glove_nn_preds) == self.dev_df["label"].values).mean()
            st_acc = (np.array(st_nn_preds) == self.dev_df["label"].values).mean() 
        
        print("\nAccuracy for Glove FFNN model: ", glove_acc)
        print("Accuracy for ST FFNN model: ", st_acc)

class Exercise4Runner:

    def __init__(self, MCQFFNN):
        self.nn_class = MCQFFNN

    def evaluate(self):

        print(
            "Running Exercise 4: Implementing Feed-Forward Neural Network for Multiple Choice Question Answering in Pytorch"
        )
        print("-----------------")

        print("Test Case 1: Testing if output shape is correct")
        ffnn = self.nn_class(
            input_dim=5
        )
        context = torch.zeros(4, 5)
        question = torch.zeros(4, 5)
        answerA = torch.zeros(4, 5)
        answerB = torch.zeros(4, 5)
        answerC = torch.zeros(4, 5)

        out = ffnn(context, question, answerA, answerB, answerC)
        print("Context Tensor: ", context)
        print("Question Tensor: ", question)
        print("Answer A Tensor: ", answerA)
        print("Answer B Tensor: ", answerB)
        print("Answer C Tensor: ", answerC)
        print("Output shape: ", out.shape)
        print("Expected Output shape: torch.Size([4, 3])")

        print("\nTest Case 2: Testing if there are correct number of parameters in the model")
        ffnn = self.nn_class(
            input_dim=5
        )
        num_params = sum(p.numel() for p in ffnn.parameters())
        print(f"Model: {ffnn}")
        print("Number of parameters: ", num_params)
        print("Expected Number of parameters: 86")
        
        print("\n Test Case 3: Test if the model is equivariant to permutation of answers")
        torch.manual_seed(42)
        ffnn = self.nn_class(
            input_dim=5
        )
        context = torch.randn(4, 5)
        question = torch.randn(4, 5)
        answerA = torch.randn(4, 5)
        answerB = torch.randn(4, 5)
        answerC = torch.randn(4, 5)
        
        out = ffnn(context, question, answerA, answerB, answerC)
        out_perm = ffnn(context, question, answerC, answerB, answerA)
        print("Output with original order (A, B, C): ", out)
        print("Output with permuted order (C, B, A): ", out_perm)


class Exercise5Runner:

    def __init__(self, train_siqa, MCQFFNN, train_data_embedded, train_labels, dev_data_embedded, dev_labels):
        self.train_siqa = train_siqa
        self.nn_class = MCQFFNN
        self.train_data_embedded = train_data_embedded
        self.train_labels = train_labels
        self.dev_data_embedded = dev_data_embedded
        self.dev_labels = dev_labels

    def evaluate(self, device="cpu"):

        print("Running Exercise 5: Training Feed-Forward Neural Network for Multiple Choice Question Answering in Pytorch")
        print("-----------------")

        torch.manual_seed(42)
        mcqffnn = self.nn_class(
            input_dim=self.train_data_embedded[0]["context"].size(0)
        )
        print("Model: ", mcqffnn)

        print("\nTraining the model")
        print("Using the following hyperparameters:")
        print("Learning Rate: 5e-5")
        print("Batch Size: 32")
        print("Number of Epochs: 20")
        print("Training Starting...")

        train_losses, dev_metrics = self.train_siqa(
            mcqffnn,
            self.train_data_embedded,
            self.train_labels,
            self.dev_data_embedded,
            self.dev_labels,
            lr=5e-5,
            batch_size=32,
            n_epochs=20,
            device=device,
            verbose=True,
        )
        print("\nTraining Completed")

        print("\nTraining Loss: ", train_losses[-1])
        print("Dev Metrics: ", dev_metrics[-1])

        return mcqffnn, train_losses, dev_metrics


class Exercise6Runner:

    def __init__(self, predict_siqa, mcqffnn_model):

        self.predict_siqa = predict_siqa
        self.model = mcqffnn_model
        self.st_model = SentenceTransformer("all-mpnet-base-v2")

        siqa_path = f"{data_dir}/socialiqa-train-dev"
        self.dev_data, self.dev_labels = load_siqa_data(siqa_path, "dev")

    def evaluate(self, device="cpu", print_q=True):

        print("Running Exercise 6: Predicting with Feed-Forward Neural Network for Multiple Choice Question Answering in Pytorch")
        print("-----------------")

        print("\nPredicting on the dev set")
        preds = self.predict_siqa(
            self.dev_data,
            self.model,
            st_model=self.st_model,
            batch_size=32,
            device=device,
        )

        if print_q:
            print("\nPrinting Predictions for first 32 questions")
            for q, pred in zip(self.dev_data[:32], preds[:32]):
                print("\nContext: ", q["context"])
                print("\nQuestion: ", q["question"])
                print("\nAnswer Choice A: ", q["answerA"])
                print("Answer Choice B: ", q["answerB"])
                print("Answer Choice C: ", q["answerC"])
                print("You Answered: ", pred)
                print("-------------------------------------------------")

        acc = (np.array(preds) == self.dev_labels).mean()
        print("\nAccuracy: ", acc)
