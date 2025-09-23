import json


def load_siqa_data(
    path,
    split,
):

    # We first load the file containing context, question and answers
    with open(f"{path}/{split}.jsonl") as f:
        data = [json.loads(jline) for jline in f.read().splitlines()]

    # We then load the file containing the correct answer for each question
    with open(f"{path}/{split}-labels.lst") as f:
        labels = f.read().splitlines()

    labels_dict = {"1": "A", "2": "B", "3": "C"}
    labels = [labels_dict[label] for label in labels]

    return data, labels
