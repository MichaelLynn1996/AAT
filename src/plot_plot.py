import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

method = ['Full fine-tuning',
'Head',
'Partial',
'Prompt fine-tuning',
'$\mathrm{AAT^{M}}$',
'$\mathrm{AAT^{MS}}$'
]

param = [87.295,
0.04,
42.552,
0.128,
3.59,
7.142,
]

acc = [95.8,
56.8,
94.1,
67.2,
98.1,
98.7,
]

s = [146, 60, 91, 92, 113, 114]

xytext = [[61, 92], [0.5, 57.4], [43, 95], [1, 67.5], [3, 94], [8 , 97]]

# markers = ['o', '+', 'x', 's', 'p', '*']

df_data = pd.DataFrame({
    'mAP': acc,
    'Params (M)': param,
    'method': method})

# sns.scatterplot(data=df_data, x='Params (M)', y='mAP', hue='method', legend=False,
#                 markers=True, style='method', size=s)
for i in range(len(s)):
    plt.scatter(param[i], acc[i], s=60)
for i in range(len(param)):
    plt.annotate(str(method[i]), xy=(param[i], acc[i]), xytext=xytext[i], fontsize=15)
    # plt.annotate(str(method[i]), xy=(param[i], acc[i]), xytext=(70, 70),
                 # arrowprops=dict(facecolor='black', shrink=0.05))

# plt.axhline(y=95.8, color='b', linestyle='-')
lw = 0.1
plt.text(40.5, 71, 'less than 8% parameters', color='r', fontsize=15, style='italic')
plt.annotate('', xy=(14, 96), xytext=(55, 75),
            arrowprops=dict(facecolor='black', shrink=0.05, lw=lw))
plt.annotate('', xy=(84, 92), xytext=(65, 75),
            arrowprops=dict(facecolor='black', shrink=0.05, lw=lw))
# 添加纵坐标网格线
plt.gca().xaxis.grid(True)
plt.xlabel('Params (M)')
plt.gca().yaxis.grid(True)
plt.ylabel('mAP')
plt.savefig('scatter.eps',dpi=600,format='eps')
plt.show()

