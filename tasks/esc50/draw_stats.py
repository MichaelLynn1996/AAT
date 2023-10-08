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

paths = ['/workspace/lh/ast-adapter/tasks/esc50/exp/test-esc50-f10-t10-impTrue-aspTrue-b36-lr1e-5',
         '/workspace/lh/ast-adapter/tasks/esc50/exp/test-esc50-f10-t10-impTrue-aspTrue-b42-lr1e-4-head_init',
         '/workspace/lh/ast-adapter/tasks/esc50/exp/test-esc50-f10-t10-impTrue-aspTrue-b42-lr1e-4-m_adapter_head_init1',
         '/workspace/lh/ast-adapter/tasks/esc50/exp/test-esc50-f10-t10-impTrue-aspTrue-b42-lr1e-4-sm_adapter_head_init1']

legends = ['Full finetune', 'Head', 'Adapter(MLP)', 'Adapter(MLP+MSA)']

for path, legend in zip(paths, legends):
    df = pd.read_csv(path + '/result.csv', header=None)
    print(df.head())

    x = list(range(0, 25))

    accs = df.iloc[:, 0].values
    plt.plot(x, accs, label=legend)

plt.xlabel("epoch")
plt.ylabel("ACC(%)")
plt.legend(loc='best', frameon=False)
plt.savefig("stats.jpg", dpi=600)
plt.close()

