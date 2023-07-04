import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import os
from matplotlib.ticker import MaxNLocator
import matplotlib
cmap = matplotlib.colormaps['tab20']

fed = True
main_path = "/data/T5/t4"
cmp1 = "fed_ll_sig_e_hyb"
cmp2 = "non"
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}

matplotlib.rc('font', **font)

# Loss Plotting
tar = "loss_train.txt"

data1 = pd.read_csv(os.path.join(main_path, cmp1, tar), header=None, names=range(9))
data2 = pd.read_csv(os.path.join(main_path, cmp2, tar), header=None, names=range(9))

plotSize = (6 * 1.618, 6)
fig, ax = plt.subplots(figsize=plotSize)
dev_size = len(data1.iloc[:, 3:].columns)
labels = ["dev"+str(i) for i in range(dev_size)]
ax.plot(data1.iloc[:, 3:], label = labels)
ax.plot(data2.iloc[:, 3:], "--",label = labels)
plt.grid(True)
# plt.legend(ncol=2)
plt.savefig(os.path.join(main_path, "loss_train.png"))
plt.close()


# F1 Plotting
#####################
#####################

tar = "f1_test.txt"
info = "info.txt"
size = 14
data1 = pd.read_csv(os.path.join(main_path, cmp1, tar), header=None, names=range(size))
# n = 1

data2 = pd.read_csv(os.path.join(main_path, cmp2, tar), header=None, names=range(size))

mask = pd.to_numeric(data2[size-1]).isnull()
data2[mask] = data2[mask].shift(axis=1)
mask = pd.to_numeric(data1[size-1]).isnull()
data1[mask] = data1[mask].shift(axis=1)

plotSize = (6 * 1.618, 6)
fig, ax = plt.subplots(figsize=plotSize)

x_mean = data1.iloc[:, 4:-1:2].values
x_std = data1.iloc[:, 5::2].values

labels = ["Set 1: dev"+str(i) for i in range(dev_size)]

for i, k in enumerate(x_mean.T):
    # x_err = 1.960 * x_std.T[i] / np.sqrt(10*n)
    
    y_plot = savgol_filter(k, 29, 3)
    ax.plot(y_plot, label = labels[i], color = cmap(i*2))
    # ax.fill_between(range(len(k)), y_plot - x_err, y_plot+ x_err, alpha=0.2)


# for i in range(3):
#     ax.fill_between(range(x_err.shape[0]), x_mean[:,i] - x_err[:,i], x_mean[:,i] + x_err[:,i], alpha=0.2)
# ax.plot(data2.iloc[:, 3:-1:2],  "--", label = labels)
x_mean = data2.iloc[:, 4:-1:2].values
x_std = data2.iloc[:, 5::2].values
labels = ["Set 2: dev"+str(i) for i in range(dev_size)]

for i, k in enumerate(x_mean.T):
    # x_err = 1.960 * x_std.T[i] / np.sqrt(10*n)
    
    y_plot = savgol_filter(k, 31, 3)
    ax.plot(y_plot, "-.",label = labels[i], color = cmap(1+i*2))

ax.set_ylabel("F1 Score")
ax.set_xlabel("Communication Rounds")
if not fed:
    ax.axvline(x = 0.4*len(y_plot), color = "black")
    plt.annotate("Global Aggregation", xy=[0.7*len(y_plot), 0.5], ha='center')
    plt.annotate("Local", xy=[0.2*len(y_plot), 0.5], ha='center')
ax.set_ylim(0,1)
plt.grid(True)
# plt.legend(ncol=2, loc=4, prop={'size': 16})
plt.tight_layout()

plt.savefig(os.path.join(main_path, "f1.pdf"))

###############################################
###############################################
###############################################
###############################################


print("FED:")

ind = 0

maxs1 = data1.iloc[ind:, 4::2].max(axis=0).values
print("maxs")
print(maxs1)
# maxs1 = data1.iloc[-1, 3::2].values
sigs1 = []
# print(maxs)
for i, v in enumerate(maxs1):
    max_ind = data1[data1[i*2+4]==v].index
    sigs1.append(data1[i*2+5][max_ind].min())
sigs1 = np.array(sigs1)

print("NON-FED:")
maxs2 = data2.iloc[ind:, 4::2].max(axis=0).values
sigs2 = []

for i, v in enumerate(maxs2):
    max_ind = data2[data2[i*2+4]==v].index
    sigs2.append(data2[i*2+5][max_ind].min())
    # print(data2[i*2+4][max_ind].min())
sigs2 = np.array(sigs2)
    
print(maxs1)
print(sigs1)
print("_____")
print(maxs2)
print(sigs2)
print("_____")
aaa = 100*(maxs1-maxs2)/maxs2
bbb = 100*(sigs1-sigs2)/sigs2


aaa1 = np.where(aaa > 0, "green", "red")
bbb1 = np.where(bbb < 0, "green", "red")
c1 = "green" if aaa.mean() > 0 else "red"
c2 = "green" if bbb.mean() < 0 else "red"


fvals = maxs1.tolist()+sigs1.tolist()+maxs2.tolist()+sigs2.tolist()+aaa1.tolist()+aaa.tolist()+bbb1.tolist()+bbb.tolist()+[c1]+[aaa.mean()]+[c2]+[bbb.mean()]
# print(fvals)
# with open("template.txt") as f:
#     f = f.read()
    
#     print(f.format(*fvals))

print(aaa)
print(bbb)
print(aaa.mean(), bbb.mean())