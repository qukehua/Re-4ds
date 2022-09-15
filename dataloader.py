from torch.utils.data import Dataset
import numpy as np
from utils import data_utils


class H36motion3D(Dataset):

    def __init__(self, path_to_data, actions, input_n=20, output_n=10, split=0, sample_rate=2):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = path_to_data
        self.split = split

        all_seqs = np.load(path_to_data)
        all_seqs = np.reshape(all_seqs, (all_seqs.shape[0], all_seqs.shape[1], -1))
        joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
        dim_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        dim_used = np.setdiff1d(np.arange(all_seqs.shape[2]), dim_ignore)

        self.all_seqs = all_seqs
        self.dim_used = dim_used
        all_seqs = all_seqs[:, :, dim_used]
        pad_idx = np.repeat([input_n - 1], output_n)
        i_idx = np.append(np.arange(0, input_n), pad_idx)

        input_dct_seq = all_seqs[:, i_idx, :]
        input_dct_seq = input_dct_seq.transpose(0, 2, 1)

        output_dct_seq = all_seqs

        self.input_dct_seq = input_dct_seq
        self.output_dct_seq = output_dct_seq

    def __len__(self):
        return np.shape(self.input_dct_seq)[0]

    def __getitem__(self, item):
        return self.input_dct_seq[item], self.output_dct_seq[item], self.all_seqs[item]