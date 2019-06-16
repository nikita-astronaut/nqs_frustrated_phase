import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

f = open(sys.argv[1], 'r')

df = pd.read_csv(sys.argv[1], sep='\t', comment='#')

df.columns = ['j2', 'train_ratio', 'test_loss', 'test_loss_err', 'test_acc', 'dtest_acc', 'train_loss', 'train_loss_err', 'train_acc', 'dtrain_acc', 'rest_loss', 'rest_loss_err', 'rest_acc', 'drest_acc', 'overlap', 'doverlap']

print(df.j2)
plt.rc('text', usetex = True)
plt.rc('font', family = 'TeX Gyre Adventor', size = 14)
plt.rc("pdf", fonttype=42)

for tr in np.unique(df['train_ratio'].values):
    df_selected = df[df['train_ratio'] == tr]

    plt.errorbar(df_selected.j2.values + 0.005, df_selected.test_acc.values, df_selected.dtest_acc.values, fmt = 'o', markersize = 5,
                         mec='black', mew=0.5, elinewidth=1.0, capsize=3.0, ecolor='black',
                         label = 'validation accuracy, tr = ' + str(tr), color = 'red')

    plt.errorbar(df_selected.j2.values, df_selected.train_acc.values, df_selected.dtrain_acc.values, fmt = 'd', markersize = 5,
                         mec='black', mew=0.5, elinewidth=1.0, capsize=3.0, ecolor='black',
                         label = 'train accuracy, tr = ' + str(tr), color = 'blue')
    plt.errorbar(df_selected.j2.values, df_selected.overlap.values, df_selected.doverlap.values, fmt = '*', markersize = 5,
                         mec='black', mew=0.5, elinewidth=1.0, capsize=3.0, ecolor='black',
                         label = 'overlap, tr = ' + str(tr), color = 'black')
plt.xlabel('$J2 / J1$')
plt.ylabel('accuracy, \\%, overlap')
plt.legend(loc = 'lower left', ncol = 1, numpoints = 1, fontsize = 12)
plt.axhline(y=0.0, color='black', linestyle='--')
plt.axhline(y=0.5, color='black', linestyle='--')
plt.grid(True)
plt.savefig('fig.pdf')
