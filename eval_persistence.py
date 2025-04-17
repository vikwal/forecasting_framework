import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import utils, eval, preprocessing


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower Simulation with Tensorflow/Keras")
    parser.add_argument('-d', '--data', type=str, help='Select dataset')
    args = parser.parse_args()
    # create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('studies', exist_ok=True)
    # read config
    config = utils.load_config('config.yaml')
    freq = config['data']['freq']
    output_dim, lag_dim, horizon = utils.handle_freq(freq=freq,
                                                     output_dim=config['model']['output_dim'],
                                                     lag_dim=config['data']['lag_dim'],
                                                     horizon=config['data']['horizon'])
    config['model']['output_dim'] = output_dim
    config['data']['horizon'] = horizon
    config['data']['lag_dim'] = lag_dim
    # load and prepare training and test data
    dfs = preprocessing.get_data(data=args.data,
                                 data_dir=config['data']['path'],
                                 freq=freq)
    # create lag features
    evaluation = pd.DataFrame()
    for key, df in tqdm(dfs.items()):
        df = utils.impute_index(data=df)
        df = preprocessing.lag_features(df=df,
                                lag_dim=lag_dim,
                                horizon=horizon,
                                lag_in_col=config['data']['lag_in_col'])
        windows = preprocessing.prepare_data(data=df,
                                            output_dim=output_dim,
                                            train_frac=config['data']['train_frac'],
                                            scale_y=config['data']['scale_y'])
        y_test = windows['y_test']
        index_test = windows['index_test']
        test_start = str(index_test[0].date())
        y_pers = eval.persistence(y=df['power'],
                                horizon=horizon,
                                from_date=test_start)
        y_pers = preprocessing.make_windows(data=y_pers,
                                            seq_len=output_dim)
        df_pers = utils.y_to_df(y=y_pers,
                    output_dim=output_dim,
                    horizon=horizon,
                    index_test=index_test,
                    t_0=None if config['eval']['eval_on_all_test_data'] else config['eval']['t_0'])
        df_true = utils.y_to_df(y=y_test,
                                output_dim=output_dim,
                                horizon=horizon,
                                index_test=index_test,
                                t_0=None if config['eval']['eval_on_all_test_data'] else config['eval']['t_0'])
        entry = eval.get_metrics(y_pred=df_pers.values,
                                         y_true=df_true.values)
        entry['key'] = [key]
        entry['output_dim'] = [output_dim]
        entry['freq'] = [freq]
        entry['t_0'] = config['eval']['t_0']
        new_evaluation = pd.DataFrame(entry)
        if evaluation.empty:
            evaluation = new_evaluation
        else:
            evaluation = pd.concat([evaluation, new_evaluation], axis=0)
    write_path = f'results/{args.data}/{args.data}_persistence.csv'
    if os.path.exists(write_path):
        evaluation.to_csv(write_path, mode='a', index=False, header=False)
    else:
        evaluation.to_csv(write_path, mode='w', index=False)

if __name__ == "__main__":
    main()