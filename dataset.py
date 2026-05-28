import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


CT_MIN = 0
CT_MAX = 100

DOSE_MIN = 0
DOSE_MAX = 80


class TLIDataset(Dataset):

    """
    Minimal dataset implementation for inference.
    """

    def __init__(
        self,
        csv_path,
        npz_dir,
        input_size=(32, 64, 64)
    ):

        self.df = pd.read_csv(csv_path)

        self.case_ids = self.df["id"].tolist()

        self.npz_dir = npz_dir

        self.input_size = input_size

        # all structured features except "id"
        self.feature_columns = [

            c for c in self.df.columns

            if c != "id"
        ]


    def normalize(
        self,
        x,
        low,
        high
    ):

        x = np.clip(
            x,
            low,
            high
        )

        x = (
            x - low
        ) / (
            high - low
        )

        return x.astype(np.float32)


    def center_crop(self, x):

        d, h, w = x.shape

        td, th, tw = self.input_size

        sd = (d - td) // 2
        sh = (h - th) // 2
        sw = (w - tw) // 2

        return x[
            sd:sd + td,
            sh:sh + th,
            sw:sw + tw
        ]


    def load_case(
        self,
        case_id
    ):

        npz_path = os.path.join(
            self.npz_dir,
            f"{case_id}.npz"
        )

        data = np.load(npz_path)

        ct = data["ct"].astype(np.float32)

        rd = data["rd"].astype(np.float32)

        rs = data["rs"].astype(np.float32)

        return ct, rd, rs


    def prepare_structured_features(
        self,
        case_id
    ):

        row = self.df[
            self.df["id"] == case_id
        ].iloc[0]

        features = row[
            self.feature_columns
        ].values.astype(np.float32)

        return features


    def __getitem__(
        self,
        index
    ):

        case_id = self.case_ids[index]

        ct, rd, rs = self.load_case(case_id)

        ct = self.center_crop(ct)
        rd = self.center_crop(rd)
        rs = self.center_crop(rs)

        ct = self.normalize(
            ct,
            CT_MIN,
            CT_MAX
        )

        rd = self.normalize(
            rd,
            DOSE_MIN,
            DOSE_MAX
        )

        image = np.stack(
            [ct, rd, rs],
            axis=0
        )

        structured_features = (
            self.prepare_structured_features(
                case_id
            )
        )

        return {

            "image": torch.tensor(
                image,
                dtype=torch.float32
            ),

            "structured_feature": torch.tensor(
                structured_features,
                dtype=torch.float32
            ),

            "id": case_id
        }


    def __len__(self):

        return len(
            self.case_ids
        )