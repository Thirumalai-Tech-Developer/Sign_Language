from torch.utils.data import Dataset
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_edge_index():
    hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
    rigth_hand = [(a + 21, b + 21) for a, b in hand_connections]
    all_edge = hand_connections + rigth_hand

    bi_direction = all_edge + [(b, a) for (a, b) in all_edge]

    return torch.tensor(bi_direction, dtype=torch.long).t().contiguous()


class HandSignSingle(Dataset):
    def __init__(self, csv_path, transform=None):
        self.csv_path = csv_path
        self.transform = transform
        self.max_frames = 0

        df = pd.read_csv(csv_path)
        if 'frame' in df.columns:
            df = df.drop(columns='frame')
        self.data = df.values.astype(np.float32)

        if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
            self.data = np.nan_to_num(self.data)

        self.max_frames = self.data.shape[0]

    def __len__(self):
        return 1
    
    def __getitem__(self, index):

        data = self.data

        #  TODO Padding

        x = torch.tensor(data, dtype=torch.float32)

        # Graph input for PyG
        edge_idx = create_edge_index().repeat(1, self.max_frames)
        batch = torch.zeros(x.size(0), dtype=torch.long)

        return x, edge_idx, batch