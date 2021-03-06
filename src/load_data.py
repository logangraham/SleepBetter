import pandas as pd
import numpy as np


def load_data(filepath=None):
    filetype = filepath.split(".")[-1]
    if filetype == 'xls':
        df = pd.read_excel(filepath)
    elif filetype == 'csv':
        df = pd.read_csv(filepath)
    else:
        raise TypeError("File must to be csv or xls")
    return df


def subset_columns(df, variable_class='all', target_column_prefix='ahi'):
    all_cols = list(df.columns)
    Y = None
    if target_column_prefix:
        target_cols = [i for i in filter(lambda x: target_column_prefix in x, all_cols)]
        Y = df[target_cols]
        df = df[[i for i in df.columns if i not in target_cols]]
    if variable_class == 'all':
        cols = df.columns
    elif variable_class == 'medical':
        med_cols = ['nrem', 'rem', 'sleepefficiency', 'arousali', 'tst50co2',
                    'zscore', 'lowsao2', 'peakc02', 'tb90']
        cols = med_cols
    elif variable_class == 'non_medical':
        non_med_cols = ['gender', 'ethnicity', 'term', 'bmi', 'age',
                        'allergies', 'asthma', 'gerd', 'tonsilsize', 'zscore']
        cols = non_med_cols
    else:
        raise ValueError("""`variable_class` has to be one of `all`, `medical',
                         or `non_medical`.""")
    return df[cols], Y

def create_targets(y, target_col='ahi', target_thresholds=[5, 10, 24]):
    targets = {}
    for thres in target_thresholds:
        name = '{}_gt_{}'.format(target_col, thres)
        s = y > thres
        if not isinstance(s, np.ndarray):
            s = np.array(s)
        targets[name] = s.astype(int)
    return targets
