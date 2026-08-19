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
    # m &= np.isin([int(i.split('_')[-1]) for i in sids],[0,5,11,16,22,27,33,38,44,49,55,60,66,71,77,82,88,93])
    m &= np.isin([int(i.split("_")[-1]) for i in sids], [0])
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
        collate_fn=collate_batch_data,
        pin_memory=True,
        prefetch_factor=2,
    )
    return dataloader


def collate_batch_data(batch_data):
    # collate features
    seq, SCOV, nn_ids, D, R, SCOD = collate_batch_features(batch_data)
    # collate labels
    y = torch.cat([data[6] for data in batch_data])
    return seq, SCOV, nn_ids, D, R, SCOD, y


def collate_batch_features(batch_data, max_num_nn=32):
    seq, SCOV = [torch.cat([data[i] for data in batch_data], dim=0) for i in range(2)]
    max_num_nn = min(max_num_nn, seq.shape[0])

    # Initialize tensors for nearest neighbors features
    num_res = sum(data[0].shape[0] for data in batch_data)
    nn_ids = torch.zeros(
        (num_res, max_num_nn), dtype=torch.long, device=batch_data[0][0].device
    )
    D = torch.zeros(
        (num_res, max_num_nn, 1), dtype=torch.float, device=batch_data[0][0].device
    )
    R = torch.zeros(
        (num_res, max_num_nn, 8), dtype=torch.float, device=batch_data[0][0].device
    )
    SCOD = torch.zeros(
        (num_res, max_num_nn, 8), dtype=torch.float, device=batch_data[0][0].device
    )

    # Cumulative size tensor
    sizes = torch.tensor([data[0].shape[0] for data in batch_data], dtype=torch.long)
    cumsum_sizes = torch.cumsum(sizes, dim=0)

    for i, (size, data) in enumerate(zip(cumsum_sizes, batch_data)):
        ix1 = size
        ix0 = ix1 - data[0].shape[0]

        # Store nearest neighbors features
        nn_ids[ix0:ix1, : data[2].shape[1]] = data[2] + ix0
        D[ix0:ix1, : data[3].shape[1], :] = data[3]
        R[ix0:ix1, : data[4].shape[1], :] = data[4]
        SCOD[ix0:ix1, : data[5].shape[1], :] = data[5]
    return seq, SCOV, nn_ids, D, R, SCOD


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_filepath):
        super(Dataset, self).__init__()
        self.dataset_filepath = dataset_filepath
        self._hf = None
        self._masked_indices = None
        with h5py.File(dataset_filepath, "r") as hf:
            self.ID = np.array(hf["metadata/ID"]).astype(np.dtype("U"))
            self.size = np.array(hf["metadata/size"])

        # default selection mask
        self.m = np.ones(len(self.ID), dtype=bool)

    def _get_file(self):
        if self._hf is None:
            self._hf = h5py.File(self.dataset_filepath, "r")
        return self._hf

    def update_mask(self, m):
        self.m &= m  # boolean vector with the size of whole dataset for masking

    def get_largest(self):
        i = np.argmax(self.size * self.m.astype(int))
        k = np.where(np.where(self.m)[0] == i)[0][0]
        return self[k]

    def __len__(self):
        return np.sum(self.m)

    def _get_masked_indices(self):
        if self._masked_indices is None:
            self._masked_indices = np.where(self.m)[0]
        return self._masked_indices

    def __getitem__(self, k):  # k is the indice of entry
        masked_indices = self._get_masked_indices()
        key_index = masked_indices[k]
        key = self.ID[key_index]

        hf = self._get_file()

        hgrp_f = hf[f"data/features/{key}"]
        hgrp_l = hf[f"data/labels/{key}"]
        # Node features
        seq = torch.tensor(np.array(hgrp_f["node/seq"]))
        SCOV = torch.tensor(np.array(hgrp_f["node/SCOV_ref"]))
        # Edge features
        D = torch.tensor(np.array(hgrp_f["edge/D_ref"]))
        R = torch.tensor(np.array(hgrp_f["edge/R_ref"]))
        SCOD = torch.tensor(np.array(hgrp_f["edge/SCOD_ref"]))
        # Edge indices
        nn_ids = torch.tensor(np.array(hgrp_f["nn_idx"]))
        # label
        labels = torch.tensor(np.array(hgrp_l["labels"]))
        return seq, SCOV, nn_ids, D, R, SCOD, labels
