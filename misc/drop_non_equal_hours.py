import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

path = '/mnt/nvme1/icon-d2/parquet/ML/'

def main():

    for root, dirs, files in tqdm(os.walk(path)):
        for file in files:
            if file.endswith(".parquet"):
                file_path = os.path.join(root, file)
                df = pd.read_parquet(file_path)
                df['starttime'] = pd.to_datetime(df['starttime'])
                hour = int(root.split('/')[-2])
                correct_hour = df[df['starttime'].dt.hour == hour]
                if len(correct_hour) != len(df):
                    write_path = os.path.join(root, file)
                    print(write_path)
                    correct_hour.to_parquet(write_path)

if __name__ == '__main__':
    main()
