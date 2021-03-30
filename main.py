# importing the libraries
import os
import pandas as pd
import numpy as np
from itertools import chain
from IPython.display import clear_output

# for reading and displaying images
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

# PyTorch libraries and modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# models
from src.models.model_util import load_model, save_checkpoint

# other files
from src.config import *
from src.training.run import *
from src.data.data_util import *

# Torch
from torch.optim import *
import torch.optim as optim
from torchsummary import summary
# from tensorboardX import SummaryWriter

# Maintenance
import mlflow
from DeepNotion.build import *

# CAM - M3dCam
# from medcam import medcam

if __name__=="__main__":

    
    cfg = load_config()

    random_seed = cfg.seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    exp_name = '3dcnn_smri_comparison_test'
    try:
        mlflow.create_experiment(name=exp_name)
    except:
        print('Existing experiment')
        
    mlflow.set_experiment(exp_name)

    #################################
    ### CHANGE CONFIGURATION HERE ###
    #################################
    cfg.model_name = 'sfcn'
    cfg.registration = 'tlrc'
    #################################
    cfg.refresh()
    model, cfg.device = load_model(cfg.model_name, verbose=False, cfg=cfg)
    print(cfg.device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    trn_dp, tst_dp = DataPacket(), DataPacket()

    run_date = today().replace('.', '_').replace(':', '') + '_' + cfg.model_name
    condition = 'TLRC Aug, 100 ep, SFCN Default'
    mlflow.start_run(run_name=condition)

    fold = None
    # cfg.epochs = 200
    db = make_db(page, client=client, schema=cfg['notion']['no_fold_aug_schema'], title='Results') if cfg['notion']['use'] else None
    for e in range(cfg.epochs):
        
        start_time = time.time()
        print(f'Epoch {e+1} / {cfg.epochs}, BEST MAE {cfg.best_mae:.3f}')
        
        model, trn_dp, trn_res = train(model, optimizer, fn_lst, trn_dp, cfg, fold=fold, augment=True)
        model, tst_dp, tst_res = valid(model, fn_lst, tst_dp, cfg, fold=fold)
        elapsed_time = round(time.time() - start_time, 3)
        
        if cfg.best_mae > tst_dp.mae[-1]:
            
            cfg.best_mae = tst_dp.mae[-1]
            model_name = f'{cfg.model_name}_ep{e}-{cfg.epochs}_sd{cfg.seed}_mae{cfg.best_mae:.3f}.pth'
            save_checkpoint(model.state_dict(), model_name, model_dir=f'./result/models/{run_date}/', is_best=True)
            
        df = pd.concat([make_df(trn_res, 'Train'),
                        make_df(tst_res, 'Test')], ignore_index=True)
        
        trn_dp.corr.update(df[df['Label'] == 'Train'].corr().Prediction['True'])
        trn_dp.refresh()
        tst_dp.corr.update(df[df['Label'] == 'Test'].corr().Prediction['True'])
        tst_dp.refresh()

        if e % 1 == 0:
            trn_dp.info('train')
            tst_dp.info('test ')

        if e % cfg.verbose_period == 0:
            # plt.title(f"L1 Losses among epochs, {e}th")
            # plt.plot(list(trn_dp.loss), label='Train')
            # plt.plot(list(tst_dp.loss), label='Test')
            # plt.grid(); plt.legend()

            # sns.lmplot(data=df, x='True', y='Prediction', hue='Label')
            # plt.grid()
            # plt.show()
            
            model_name = f'{cfg.model_name}_ep{e}-{str(cfg.epochs).zfill(3)}_sd{cfg.seed}_mae{cfg.best_mae:.3f}.pth'
            save_checkpoint(model.state_dict(), model_name, model_dir=f'./result/models/{run_date}/', is_best=False)
            
            if db:
                data = gather_data(e=e, time=elapsed_time, cfg=cfg,
                                train=trn_dp, valid=tst_dp)
                write_db(db, data)
        
        metrics = mlflow_data(time=elapsed_time, train=trn_dp, valid=tst_dp)
        mlflow.log_metrics(metrics, e)
        
        torch.cuda.empty_cache()
        
    # Save Parameters to MLFlow
    cfg.best_mae = min(tst_dp.mae)
    cfg.refresh()
    params = dict()
    for name, value in cfg.get_dict().items():
        if name not in ['notion']:  
            params[name] = str(value)
    mlflow.log_params(params)

    save_checkpoint(cfg.get_dict(), 'cfg.pt', model_dir=f'./result/models/{run_date}/', is_best=True)

    # Save Plots to MLFlow
    sns.jointplot(data=df[df['Label'] == 'Test'], x='Prediction', y='True', kind='reg')
    plt.grid()
    plt.savefig(f'./result/models/{run_date}/test_jointplot.png')
    plt.show()
    # mlflow.log_artifact(f'./result/models/{run_date}/test_jointplot.png')
    plt.close()

    plt.title(f"L1 Losses\n{condition}")
    plt.plot(list(trn_dp.loss), label='Train')
    plt.plot(list(tst_dp.loss), label='Test')
    plt.grid(); plt.legend()
    plt.savefig(f'./result/models/{run_date}/loss_plot.png')
    plt.show()
    # mlflow.log_artifact(f'./result/models/{run_date}/loss_plot.png')

    mlflow.end_run()