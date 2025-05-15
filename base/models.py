from django.db import models
import torch
import torch_geometric.nn as gnn
import torch.nn as nn
import mediapipe as mp
import cv2
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mp_hand = mp.solutions.hands
hands = mp_hand.Hands(static_image_mode=False, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

import cv2
import csv
import pandas as pd
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def GraphFramemrk(video_path, out_csv):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    with open(out_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        header = ['frame']
        for hand_idx in range(2):
            for point in range(21):
                header += [f'hand{hand_idx}_x{point}', f'hand{hand_idx}_y{point}', f'hand{hand_idx}_z{point}']
        writer.writerow(header)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            row = [frame_idx]
            hand_landmarks = results.multi_hand_landmarks or []
            for h in range(2):
                if h < len(hand_landmarks):
                    for lm in hand_landmarks[h].landmark:
                        row += [lm.x, lm.y, lm.z]
                else:
                    row += [None] * 63

            writer.writerow(row)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    df = pd.read_csv(out_csv)

    landmark_columns = df.columns.drop('frame')
    df = df.dropna(how='all', subset=landmark_columns)

    df.to_csv(out_csv, index=False)


class GTv1GraphConfig():
    def __init__(
            self,
            input_shape=126,
            output_shape=2000,
            hidden_size=512,
            num_class=2000,
            intermediate_size=2048,
            eps=1e-8,
    ):
        self.hidden_size = hidden_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_class = num_class
        self.intermediate_size = intermediate_size
        self.eps = eps


class GTv1RMSNorm(nn.Module):
    def __init__(self, config: GTv1GraphConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.output_shape)).to(device)
        self.eps = config.eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32).to(device)
        variance = hidden_states.pow(2).mean(-1, keepdim=True).to(device)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps).to(device)
        hidden_states = hidden_states * self.weight.to(dtype=input_dtype, device=device)

        return hidden_states


class GTv1GraphNN(nn.Module):
    def __init__(self, config: GTv1GraphConfig):
        super().__init__()
        self.gc1 = gnn.GCNConv(config.input_shape, config.hidden_size).to(device)
        self.gc2 = gnn.GCNConv(config.hidden_size, config.intermediate_size).to(device)
        self.gc3 = gnn.GCNConv(config.intermediate_size, config.output_shape).to(device)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.SiLU()

    def forward(self, hidden_states, edge_index):
        hidden_states = self.gc1(hidden_states, edge_index)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.gc2(hidden_states, edge_index)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.gc3(hidden_states, edge_index)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class GTv1GraphModel(nn.Module):
    def __init__(self, config: GTv1GraphConfig):
        super().__init__()
        self.norm = GTv1RMSNorm(config).to(device)
        self.graph_nn = GTv1GraphNN(config).to(device)
        self.pool = gnn.global_mean_pool
        self.classifier = nn.Linear(config.output_shape, config.num_class).to(device)

    def forward(self, hidden_states, edge_index, batch):
        hidden_states = self.graph_nn(hidden_states, edge_index)
        hidden_states = self.norm(hidden_states)
        pooled = self.pool(hidden_states, batch)
        return self.classifier(pooled)

config = GTv1GraphConfig()
model = GTv1GraphModel(config)