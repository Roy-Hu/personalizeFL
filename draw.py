import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import  matplotlib.font_manager

plt.rcParams['font.family'] = 'Times New Roman'

xlabel_fontsize = 28
ylabel_fontsize = 28
xticks_fontsize = 24
yticks_fontsize = 24

path = 'exp/'
pattern = '.csv'
avg_files_num = 3
with_folder = True
title = np.array(['CoPT', 'Train_Acc', 'Test_Acc', 'Best_Train_Acc', 'Best_Test_Acc', 'Reward'])
color = ['#00AFAF', '#00AF00', '#AF0000']

def find(pattern, path, with_folder=True):
    rtn = []
    if with_folder:
        folders = os.listdir(path)
        for folder in folders:
            files_list = []
            for files in os.listdir(path + folder):
                if pattern in files:
                    files_list.append(os.path.join((path + folder + '/'), files))
            rtn.append(files_list)
    else:
        for root, dirs, files in os.walk(path):
            if pattern in files:
                rtn.append(os.path.join(root, pattern))
    return rtn

def draw():
    plt.xlim(330, 400)
    plt.xticks(np.arange(340, 400, step=12), fontsize=xticks_fontsize)
    plt.ylim(0.45, 0.65)
    plt.yticks(np.arange(0.45, 0.7, step=0.05), fontsize=yticks_fontsize)

    ymajorFormatter = ticker.FormatStrFormatter('%3.2f') #设置y轴标签文本的格式
    ax.yaxis.set_major_formatter(ymajorFormatter)
    
    leg = plt.legend(
        [r'$\mathcal{Z}$' + ' = 8', r'$\mathcal{Z}$' + ' = 16'],
        loc=10,
        bbox_to_anchor=(0.5, 1.12),\
        prop={'size': '21', 'family': 'Times New Roman'},
        frameon='False',
        labelspacing=0,
        handletextpad=0.2,
        handlelength=1,
        columnspacing=0.5,
        ncol=2,
        facecolor='None'
    )
    leg.get_frame().set_linewidth(0.0)

    plt.ylabel('Accuracy', fontsize=ylabel_fontsize)
    plt.xlabel('CoPT(' + r'$\times$' + '100)', fontsize=xlabel_fontsize)
    plt.tight_layout()

    loc = "./HT_accuracy.pdf"
    plt.savefig(loc)
    loc = "./HT_accuracy.png"
    plt.savefig(loc, bbox_inches = 'tight')


if __name__ == "__main__":
    files_list = find(pattern, path, with_folder)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    for f_idx, files in enumerate(files_list):
        if len(files) >= avg_files_num:
            files_data = []
            for file_name in files:
                with open(file_name, 'r') as f:
                    data = f.readlines()
                for idx, d in enumerate(data):
                    data[idx] = d.split('\n')[0].split(',')
                files_data.append(data)
            files_data = np.mean(np.array(files_data, dtype='float'), axis=0)

            x = (files_data[:, np.where('CoPT' == title)[0][0]] + 4000) / 100
            # x = [*range(700)]
            y = files_data[:, np.where('Best_Test_Acc' == title)[0][0]]

            plt.plot(x, y, color=color[f_idx], lw=1.0, label='HT = 28', linestyle='-', markerfacecolor='none')
    draw()
