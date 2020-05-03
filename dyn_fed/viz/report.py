"""Visualization for report analysis
"""
import matplotlib.pyplot as plt

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
