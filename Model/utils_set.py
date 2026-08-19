import h5py
import numpy as np
import torch

from architecture.config import config_runtime


def setup_dataloader(config_data, sids_selection_filepath):
    sids_sel = np.genfromtxt(sids_selection_filepath, dtype=np.dtype("U"))
    dataset = Dataset(config_data["dataset_filepath"])
    sids = np.array([key for key in dataset.ID])
    m = np.isin([i.rsplit("_", 1)[0] for i in sids], sids_sel)
    # m = np.isin([i.split('_')[0]+'_'+i.split('_')[1] for i in sids], sids_sel)
    # m = np.isin([i.split('_')[0] for i in sids], sids_sel)
    # Further filter out big ones
    sizes = np.array([s for s in dataset.size])
    big_ones = np.where(sizes > config_data["max_size"])
    m[big_ones] = False
    dataset.update_mask(m)
    # define data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config_runtime["batch_size"],
        shuffle=True,
        num_workers=8,
        collate_fn=collate_set_transformer,
        pin_memory=True,
        prefetch_factor=2,
    )
    return dataloader


def collate_set_transformer(batch):
    # batch is a list containing one item: [(seq, [scov...], [nn_ids...], ...)]
    # We just want to return the inner tuple.
    return batch[0]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_filepath):
        super(Dataset, self).__init__()
        self.dataset_filepath = dataset_filepath
        self._hf = None
        # self.conformations = [0, 5, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        self.conformations = list(range(25))

        with h5py.File(dataset_filepath, "r") as hf:
            self.ID = np.array(hf["metadata/ID"]).astype(np.dtype("U"))
            self.size = np.array(hf["metadata/size"])
        # Get the base protein ID for each entry (e.g., '1a2b' from '1a2b_44')
        self.base_ids = np.array(
            [i.rsplit("_", 1)[0] for i in self.ID]
        )  # np.array([i.split('_')[0]+'_'+i.split('_')[1] for i in self.ID])

        # default selection mask
        self.m = np.ones(len(self.ID), dtype=bool)
        self.masked_protein_ids = np.unique(self.base_ids)

    def _get_file(self):
        if self._hf is None:
            self._hf = h5py.File(self.dataset_filepath, "r")
        return self._hf

    def update_mask(self, m):
        self.m &= m
        self.masked_protein_ids = np.unique(self.base_ids[self.m])

    def get_largest(self):
        masked_idx = np.where(self.m)[0]
        masked_sizes = self.size[masked_idx]
        i = masked_idx[np.argmax(masked_sizes)]
        protein_id = self.base_ids[i]
        k = np.where(self.masked_protein_ids == protein_id)[0][0]
        return self[k]

    def __len__(self):
        return len(self.masked_protein_ids)

    def __getitem__(self, k):  # # k index of unique proteins
        protein_id = self.masked_protein_ids[k]
        scov_list, nn_ids_list, d_list, r_list, scod_list = [], [], [], [], []
        labels = None
        hf = self._get_file()
        for conf in self.conformations:
            key = f"{protein_id}_{conf}"

            # Check if this conformation exists
            if f"data/features/{key}" not in hf:
                continue
            hgrp_f = hf[f"data/features/{key}"]

            # Append features for this conformation to our lists
            scov_list.append(torch.tensor(np.array(hgrp_f["node/SCOV_ref"])))
            d_list.append(torch.tensor(np.array(hgrp_f["edge/D_ref"])))
            r_list.append(torch.tensor(np.array(hgrp_f["edge/R_ref"])))
            scod_list.append(torch.tensor(np.array(hgrp_f["edge/SCOD_ref"])))
            nn_ids_list.append(torch.tensor(np.array(hgrp_f["nn_idx"])))

            # Load label and seq only once
            if labels is None:
                seq = torch.tensor(np.array(hgrp_f["node/seq"]))
                hgrp_l = hf[f"data/labels/{key}"]
                labels = torch.tensor(np.array(hgrp_l["labels"]))

        return seq, scov_list, nn_ids_list, d_list, r_list, scod_list, labels
