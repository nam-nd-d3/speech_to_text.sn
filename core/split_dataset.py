import os
import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Split datasets wav2vec2.')
parser.add_argument('--root_data_csv',
                    type=str,
                    required=True,
                    help="str - csv directory path : type (path_file, transcripts)")

parser.add_argument('--train_data_csv',
                    type=str,
                    required=True,
                    help="str - csv directory path : type (path_file, transcripts)")

parser.add_argument('--test_data_csv',
                    type=str,
                    required=True,
                    help="str - csv directory path : type (path_file, transcripts)")

parser.add_argument('--ratio_split',
                    default=0.8,
                    type=float,
                    required=True,
                    help="str - ratio to split dataset")

args = parser.parse_args()

root_dataset = pd.read_csv(args.root_data_csv)

msk = np.random.rand(len(root_dataset)) <= args.ratio_split
train_csv = root_dataset[msk]
test_csv = root_dataset[~msk]
train_csv.to_csv(args.train_data_csv)
test_csv.to_csv(args.test_data_csv)
print(f"Train size: {len(train_csv)} - Test size: {len(test_csv)}")
