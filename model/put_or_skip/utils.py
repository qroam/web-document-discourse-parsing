import torch

from utils import father_id_to_previous_id, common_collate_fn

def putorskip_collate_fn(batch):

    output = common_collate_fn(batch)
    
    assert len(batch) == 1  # TODO
    num_paragraph = len(batch[0]["input_ids"])

    father_labels = [instance["Father"] for instance in batch]
    # previous_labels = [instance["Previous_Relation_ids"] for instance in batch]

    previous_node_ids = father_id_to_previous_id([[idx + 1 for idx in father_ids] for father_ids in father_labels])
    # previous_labels = [[idx if idx!=3 else 1 for idx in instance["Previous_Relation_ids"]] for instance in batch]  # abolish the "NA" type in previous relations, replacing by "Break"
    # previous_labels = [[idx if (idx != 3 or previous_node_ids[i][j + 1] == 0) else 1 for j, idx in enumerate(lst)] for i, lst in enumerate(previous_labels)]


    """input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)"""
    # print(father_labels)
    father_labels = torch.tensor(father_labels, dtype=torch.long)
    # previous_labels = torch.tensor(previous_labels, dtype=torch.long)
    upper_triangluar_padding = torch.ones(num_paragraph + 1, num_paragraph + 1) - torch.tril(
        torch.ones(num_paragraph + 1, num_paragraph + 1))
    previous_node_ids = torch.tensor(previous_node_ids, dtype=torch.long)
    # labels = torch.tensor(labels, dtype=torch.long)
    # ss = torch.tensor(ss, dtype=torch.long)
    # os = torch.tensor(os, dtype=torch.long)

    # output = (input_ids, input_mask, father_labels, previous_labels)
    training_tuple_ids = [instance["training_tuple_ids"] for instance in batch]


    output.update({
        "training_tuple_ids": training_tuple_ids,
    })
    return output

