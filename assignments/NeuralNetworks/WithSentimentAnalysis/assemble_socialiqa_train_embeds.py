import torch
import os

components = []
for i in range(1, 7):
    components.append(torch.load(f"data/socialiqa-train-dev/train_data_embedded.part{i}.pt", weights_only=True))

def make_dict_out_of_fivecol_tensor(fivecol_tensor):
    dict_to_return = {}
    col_names = ['context', 'question', 'answerA', 'answerB', 'answerC']
    for i in range(5):
        dict_to_return[col_names[i]] = fivecol_tensor[i]
    return dict_to_return

if isinstance(components[0], torch.Tensor):
    t = torch.cat(components)
    torch.save(t, "data/socialiqa-train-dev/train_data_embedded.pt")
elif isinstance(components[0], list):
    overall_list = []
    for component in components:
        overall_list += [make_dict_out_of_fivecol_tensor(fivecol_tensor) for fivecol_tensor in component]
    torch.save(overall_list, "data/socialiqa-train-dev/train_data_embedded.pt")
else:
    assert False, "datatype of individual train_data_embedded components not recognized"

for i in range(1, 7):
    os.remove(f"data/socialiqa-train-dev/train_data_embedded.part{i}.pt")
