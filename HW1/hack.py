# import argparse


# def train():
#     print("Training...")


# def inference():
#     print("Inference...")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train or perform inference")
#     parser.add_argument(
#         "--mode",
#         action="store_true",
#         choices=["train", "inference"],
#         help="Mode: train or inference",
#     )
#     args = parser.parse_args()

#     if args.train:
#         confirmation = input(
#             "Are you sure you want to perform model training? This will overwrite existing model. (yes/no): "
#         )
#         if confirmation.lower() == "yes":
#             train()
#         else:
#             print("Training aborted.")
#     elif args.mode == "inference":
#         inference()

import argparse


def train():
    print("Training...")


def inference():
    print("Inference...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or perform inference")
    # parser.add_argument("--train", action="store_true", help="Perform model training")
    args = parser.parse_args()

    confirmation = input(
        "Are you sure you want to perform model training? This will overwrite existing model. (train/inf): "
    )
    if confirmation.lower() == "train":
        train()
    elif confirmation.lower() == "inf":
        inference()
    else:
        print("Error op.")
