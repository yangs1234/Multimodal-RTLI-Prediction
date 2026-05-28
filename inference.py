import os
import torch

from torch.utils.data import DataLoader

from dataset import TLIDataset
from model import TLINet


# -------------------------
# Inference configuration
# -------------------------

CSV_PATH = "./sample_data/sample.csv"

NPZ_DIR = "./sample_data/"

CHECKPOINT_PATH = (
    "./checkpoints/pretrained_model.pth"
)

INPUT_SIZE = (32, 64, 64)

BATCH_SIZE = 1

NUM_WORKERS = 0

NUM_CLASSES = 60

N_STRUCTURED_FEATURES = 12


def main():

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )

    # -------------------------
    # Dataset
    # -------------------------

    dataset = TLIDataset(

        csv_path=CSV_PATH,

        npz_dir=NPZ_DIR,

        input_size=INPUT_SIZE
    )


    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS
    )

    # -------------------------
    # Model
    # -------------------------

model = TLINet(

    num_classes=NUM_CLASSES,

    structured_dim=(
        N_STRUCTURED_FEATURES
    )
)

    # -------------------------
    # Optional checkpoint loading
    # -------------------------

    if os.path.exists(
        CHECKPOINT_PATH
    ):

        checkpoint = torch.load(

            CHECKPOINT_PATH,

            map_location=device
        )

        model.load_state_dict(
            checkpoint
        )

        print(
            "Pretrained weights loaded."
        )

    else:

        print(

            "No pretrained weights found. "

            "Using randomly initialized model."
        )

    model = model.to(device)

    model.eval()

    # -------------------------
    # Inference
    # -------------------------

    with torch.no_grad():

        for batch in loader:

            image = batch[
                "image"
            ].to(device)

            structured_feature = batch[
                  "structured_feature"
            ].to(device)

            pred = model(

                image,

                structured_feature
            )

            print(

                "Prediction shape:",

                pred.shape
            )


if __name__ == "__main__":

    main()