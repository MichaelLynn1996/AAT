#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Time         : 2023/6/2 12:38
# @Author       : Hai Lin
# @Affiliation  : South China Agricultural University
# @Email        : sealynndev@gmail.com
# @File         : draw_stats.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

paths = ['/workspace/lh/ast-adapter/tasks/speechcommands/exp/test-speechcommands-f10-t10-pTrue-b128-lr2.5e-4',
         '/workspace/lh/ast-adapter/tasks/speechcommands/exp/test-speechcommands-f10-t10-pTrue-b128-lr2.5e-4-head',
         '/workspace/lh/ast-adapter/tasks/speechcommands/exp/test-speechcommands-f10-t10-pTrue-b128-lr2.5e-4-m_adapter_head',
         '/workspace/lh/ast-adapter/tasks/speechcommands/exp/test-speechcommands-f10-t10-pTrue-b128-lr2.5e-4-sm_adapter_head']

legends = ['Full finetune', 'Head', 'Adapter(MLP)', 'Adapter(MLP+MSA)']

for path, legend in zip(paths, legends):
    df = pd.read_csv(path + '/result.csv', header=None)
    # print(df.head())

    x = list(range(0, 30))

    accs = df.iloc[:, 0].values
    plt.plot(x, accs, label=legend)

plt.xlabel("epoch")
plt.ylabel("ACC(%)")
plt.legend(loc='best', frameon=False)
plt.savefig("stats.jpg", dpi=600)
plt.close()

