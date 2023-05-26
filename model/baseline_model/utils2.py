import torch

from utils import father_id_to_previous_id, common_collate_fn


def collate_fn(batch):

    output = common_collate_fn(batch)

    assert len(batch) == 1  # TODO
    num_paragraph = len(batch[0]["input_ids"])

    golden_parent_ids = [instance["Father"] for instance in batch]
    golden_parent_ids = torch.tensor(golden_parent_ids, dtype=torch.long)

    golden_parent_labels = [instance["Father_Relation_ids"] for instance in batch] 
    golden_parent_labels = torch.tensor(golden_parent_labels, dtype=torch.long)
    
    output.update(
      {
        "golden_parent_ids": golden_parent_ids,
        "golden_parent_labels": golden_parent_labels,
      }
      )
    return output

