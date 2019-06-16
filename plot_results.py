import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

f = open(sys.argv[1], 'r')

df = pd.read_csv(sys.argv[1], sep='\t', comment='#')

df.columns = ['j2', 'test_loss', 'dtest_loss', 'test_acc', 'dtest_acc', 'rest_loss', 'drest_loss', 'rest_acc', 'drest_acc', 'overlap', 'doverlap']

plt.rc('text', usetex = True)
plt.rc('font', family = 'TeX Gyre Adventor', size = 14)
plt.rc("pdf", fonttype=42)

plt.errorbar(df.j2.values + 0.005, df.test_acc.values, df.dtest_acc.values, fmt = 'o', markersize = 5,
                         mec='black', mew=0.5, elinewidth=1.0, capsize=3.0, ecolor='black',
                         label = 'train accuracy', color = 'red')

plt.errorbar(df.j2.values, df.rest_acc.values, df.drest_acc.values, fmt = 'd', markersize = 5,
                         mec='black', mew=0.5, elinewidth=1.0, capsize=3.0, ecolor='black',
                         label = 'validation accuracy', color = 'blue')
plt.errorbar(df.j2.values, df.overlap.values, df.doverlap.values, fmt = '*', markersize = 5,
                         mec='black', mew=0.5, elinewidth=1.0, capsize=3.0, ecolor='black',
                         label = 'overlap', color = 'black')
plt.xlabel('$J2 / J1$')
plt.ylabel('accuracy, \\%, overlap')
plt.legend(loc = 'lower left', ncol = 1, numpoints = 1, fontsize = 12)
plt.axhline(y=0.0, color='black', linestyle='--')
plt.axhline(y=0.5, color='black', linestyle='--')
plt.grid(True)
plt.savefig('fig.pdf')
