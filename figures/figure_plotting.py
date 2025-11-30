import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Classes.config import Config

def plot_accuracy_comparison():
    classifiers = ['KNN', 'Linear SVM', 'RBF SVM', 'Random Forest']

    psd_dict = Config.metrics_queue_psd.get()
    csp_dict = Config.metrics_queue_csp.get()

    psd_all = [
        psd_dict['KNN']['accuracy'],
        psd_dict['SVM_Linear']['accuracy'],
        psd_dict['SVM_RBF']['accuracy'],
        psd_dict['Random_Forest']['accuracy']
    ]

    csp_all = [
        csp_dict['KNN']['accuracy'],
        csp_dict['SVM_Linear']['accuracy'],
        csp_dict['SVM_RBF']['accuracy'],
        csp_dict['Random_Forest']['accuracy']
    ]

    print(f"PSD Accuracies: {psd_all}")
    print(f"CSP Accuracies: {csp_all}")

    barWidth = 0.2
    r1 = np.arange(len(classifiers))
    r2 = [x + barWidth for x in r1]

    psd_color = '#1f77b4'
    csp_color = '#2ca02c'

    plt.figure(figsize=(14, 8))
    plt.rcParams.update({'font.size': 14})

    bars1 = plt.bar(r1, psd_all, width=barWidth, color=psd_color,
                    edgecolor='white', label='All PSD')
    bars2 = plt.bar(r2, csp_all, width=barWidth, color=csp_color,
                    edgecolor='white', label='All CSP')

    for i in range(len(classifiers)):
        plt.axvline(x=(r1[i] + r2[i]) / 2, color='gray',
                    linestyle='--', alpha=0.3)

    for i in range(len(classifiers)):
        mid_point = (r1[i] + r2[i]) / 2
        plt.annotate(
            classifiers[i],
            xy=(mid_point, 0.76),
            xytext=(0, -30),
            textcoords="offset points",
            ha='center',
            fontweight='bold',
            fontsize=16
        )

    plt.ylabel('Accuracy', fontweight='bold', fontsize=16)
    plt.title('Accuracy Comparison Across Different Feature Extraction Methods',
              fontweight='bold', fontsize=18)

    plt.xticks([])

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.005,
                f'{height:.4f}',
                ha='center',
                va='bottom',
                fontsize=11
            )

    add_labels(bars1)
    add_labels(bars2)

    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    # plt.ylim(0.75, 0.95)
    plt.yticks(fontsize=14)

    legend_elements = [
        Patch(facecolor=psd_color, edgecolor='white', label=f'{Config.mode} PSD'),
        Patch(facecolor=csp_color, edgecolor='white', label=f'{Config.mode} CSP'),
    ]
    plt.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.07),
        ncol=4,
        frameon=True,
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f'{Config.mode}_PSD_CSP_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
