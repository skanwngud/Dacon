import numpy as np
import pandas as pd

def losses():
    loss1=pd.read_csv('./dacon/quantile_loss_0.1.csv', index_col=0, header=0)
    loss2=pd.read_csv('./dacon/quantile_loss_0.2.csv', index_col=0, header=0)
    loss3=pd.read_csv('./dacon/quantile_loss_0.3.csv', index_col=0, header=0)
    loss4=pd.read_csv('./dacon/quantile_loss_0.4.csv', index_col=0, header=0)
    loss5=pd.read_csv('./dacon/quantile_loss_0.5.csv', index_col=0, header=0)
    loss6=pd.read_csv('./dacon/quantile_loss_0.6.csv', index_col=0, header=0)
    loss7=pd.read_csv('./dacon/quantile_loss_0.7.csv', index_col=0, header=0)
    loss8=pd.read_csv('./dacon/quantile_loss_0.8.csv', index_col=0, header=0)
    loss9=pd.read_csv('./dacon/quantile_loss_0.9.csv', index_col=0, header=0)

    losses=pd.concat([loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8, loss9], axis=1)

    losses.to_csv('./dacon/losses.csv')
    return losses

print(losses.info())