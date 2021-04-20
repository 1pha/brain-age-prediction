import os
import pandas as pd

def stratified_sample_df(df, col, n_samples):
    '''
    For stratified sampling by columns
    Just put -
        df       : pandas.DataFrame
        col      : string of the column,
        n_samples: the number of samples you want from each column category
    '''
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n, random_state=42))
    df_.index = df_.index.droplevel(0)
    return df_

label = pd.read_csv('../rsc/age_ixidlbsoas13.csv', index_col=0)
def path_maker(row):

    brain_id = row.id
    src = row.src
    
    if src == 'Oasis3':
        SUFFIX = '.nii-brainmask.nii'
        
    else:
        SUFFIX = '-brainmask.nii'

    ROOT = '../../brainmask_nii/'
    path = ROOT + brain_id + SUFFIX
    return path if os.path.exists(path) else brain_id

FNAMES = stratified_sample_df(label, 'src', 3).apply(path_maker, axis=1).values