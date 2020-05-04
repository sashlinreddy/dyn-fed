"""Visualization for report analysis
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_results(results,
                 metric,
                 n_rows=2,
                 n_cols=2,
                 xlabel='',
                 ylabel='',
                 suptitle='',
                 titles=None,
                 kind='line',
                 title_fontsize=14,
                 figsize=(12, 8)):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True, sharex=True)
    # add a big axes, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    for i in range(n_rows):
        for j in range(n_cols):
            row = i * n_cols + j
            if kind == 'bar':
                results[row][metric].plot(
                    kind=kind,
                    ax=axes[i, j],
                    width=0.85,
                    rot=0
                )
            else:
                results[row][metric].plot(
                    kind=kind,
                    marker='o',
                    markersize=2,
                    ax=axes[i, j]
                )
            axes[i, j].set_title(
                titles[row],
                fontdict=dict(fontsize=title_fontsize)
            )
            axes[i, j].set_xlabel('')
            if (i != (n_rows - 1)) or (j != (n_cols - 1)):
                axes[i, j].legend().remove()

    handles, labels = axes[n_rows - 1, n_cols - 1].get_legend_handles_labels()
    axes[n_rows - 1, n_cols - 1].legend().remove()
    # fig.legend(handles, labels, loc='upper right', fontsize=12)
    fig.legend(handles, labels, loc=7, fontsize=11)
    # fig2.tight_layout()
    fig.subplots_adjust(right=0.85)

    plt.setp(axes[1, 0].get_xticklabels(), fontsize=11)
    plt.setp(axes[1, 1].get_xticklabels(), fontsize=11)
    plt.setp(axes[0, 1].get_yticklabels(), fontsize=11)
    plt.setp(axes[1, 1].get_yticklabels(), fontsize=11)
    plt.ylabel(ylabel, fontdict=dict(fontsize=14))
    plt.xlabel(xlabel, fontdict=dict(fontsize=14))
    plt.suptitle(suptitle, fontsize=18)
    # plt.savefig('../reports/figures/time-analysis.png', dpi=500, format='png')

def plot_pkt_size(df1, df2, figsize=(14, 6)):
    """Plot packet size
    """
    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=figsize)
    # add a big axes, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    # Mnist
    df1.plot(kind='bar', ax=axes[0], width=0.8)
    axes[0].set_title('MNIST', fontdict=dict(fontsize=16))
    axes[0].set_xlabel('')
    axes[0].legend().remove()
    # Fashion Mnist
    df2.plot(kind='bar', ax=axes[1], width=0.8)
    axes[1].set_title('Fashion-MNIST', fontdict=dict(fontsize=16))
    axes[1].set_xlabel('')
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend().remove()
    # fig.legend(handles, labels, loc='upper right', prop={'size': 12})
    fig.legend(handles, labels, loc='lower center', fontsize=11, ncol=4)
    # fig2.tight_layout()
    fig.subplots_adjust(bottom=0.25)


    plt.suptitle('Packet size analysis', fontsize=18)
    plt.xlabel('No. of clients', fontdict=dict(fontsize=14))
    plt.ylabel('Packet size (MB)', fontdict=dict(fontsize=14))
    plt.setp(axes[0].get_xticklabels(), ha='center', rotation=0, fontsize=12)
    plt.setp(axes[1].get_xticklabels(), ha='center', rotation=0, fontsize=12)
    plt.setp(axes[0].get_yticklabels(), fontsize=12)
    plt.setp(axes[1].get_yticklabels(), fontsize=12)
    # plt.savefig('../reports/figures/packet-size-analysis.png', dpi=500, format='png')

def plot_model_performance(results,
                           metric1='test_acc',
                           metric2='gen_gap',
                           xlabel='No. of clients',
                           ylabel1='Accuracy',
                           ylabel2='Generalization gap',
                           suptitle='Model Performance',
                           ylim2=(-0.05, 0.2),
                           width=0.1,
                           titles=None,
                           figsize=(12, 8)):
    """Plot model performance
    """
    nrows = 2
    ncols = 2
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharey=True,
        sharex=True
    )
    # fig.add_subplot(111, frameon=False)
    # # hide tick and tick label of the big axes
    # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    for i in np.arange(nrows):
        for j in np.arange(ncols):
            row = i * ncols + j
            # q = 3
            x = np.arange(len(results[row][metric2].index))[0:]
            bars = np.arange(len(results[row][metric2].columns))
            min_x = x - width / 2
            max_x = (x + width * len(bars))
            multiplier = 10 ** 1
            ticks = np.floor(((max_x - min_x) / 2 + min_x) * multiplier) / multiplier
            lines = axes[i][j].plot(
                ticks,
                results[row][metric1].values,
                linestyle='--',
                linewidth=1.5,
                marker='o',
                markersize=3
            )
            axes[i][j].set_ylim(0, 100)
            axes[i][j].set_title(titles[row], fontdict=dict(fontsize=14))
            ax2 = axes[i][j].twinx()
            for k in bars:
                if k == 0:
                    r = x.copy()
                else:
                    r = [b + width for b in r]
                ax2.bar(r, results[row][metric2].values[:, k], width=width, alpha=0.4)

            # ax.set_xticks([r + width**1/(1/1) for r in range(len(x))])
            ax2.set_xticks(np.floor(((max_x - min_x) / 2 + min_x) * multiplier) / multiplier)
            ax2.set_xticklabels(results[row][metric2].index)
            if j == 0:
                ax2.set_yticklabels([])
            ax2.set_ylim(*ylim2)
            axes[i, j].set_xlabel('')
            ax2.set_xlabel('')

    axes1 = fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            
    fig.subplots_adjust(right=0.80)
    fig.legend(lines, results[-1][metric2].columns, loc=7, fontsize=11)
    plt.setp(axes[1, 0].get_xticklabels(), fontsize=11)
    plt.setp(axes[1, 1].get_xticklabels(), fontsize=11)
    plt.setp(axes[0, 1].get_yticklabels(), fontsize=11)
    plt.setp(axes[1, 1].get_yticklabels(), fontsize=11)
    plt.ylabel(ylabel1, fontdict=dict(fontsize=14))
    # axes2.set_ylabel(ylabel2, fontdict=dict(fontsize=14))
    plt.xlabel(xlabel, fontdict=dict(fontsize=14))
    plt.suptitle(suptitle, fontsize=18)
    axes2 = axes1.twinx()
    axes2.set_yticks([])
    axes2.set_yticklabels([])
    # axes2.axis('off')
    axes2.spines['top'].set_visible(False)
    axes2.spines['right'].set_visible(False)
    axes2.spines['bottom'].set_visible(False)
    axes2.spines['left'].set_visible(False)
    axes2.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    axes2.set_ylabel(ylabel2, fontdict=dict(fontsize=14), labelpad=33)
