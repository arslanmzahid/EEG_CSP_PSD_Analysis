import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Define classifiers and their accuracy values for each approach
classifiers = ['KNN', 'Linear SVM', 'RBF SVM', 'Random Forest']

# Accuracy values for each approach and classifier
psd_all = [0.8276, 0.8719, 0.8867, 0.8424]  # All PSD features
psd_sel = [0.8276, 0.8571, 0.8621, 0.8424]  # Selected PSD features
csp_all = [0.9163, 0.8276, 0.9015, 0.8916]  # All CSP components
csp_sel = [0.8916, 0.8079, 0.9113, 0.8719]  # Selected CSP components

# Set width of bars
barWidth = 0.2
r1 = np.arange(len(classifiers))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# Use a cleaner color palette with only two base colors (PSD vs CSP)
psd_color = '#1f77b4'  # Blue for PSD
csp_color = '#2ca02c'  # Green for CSP

# Create the plot with larger overall size
plt.figure(figsize=(14, 8))

# Increase the font size for all text elements
plt.rcParams.update({'font.size': 14})

# Create bars with hatches for better distinction
bars1 = plt.bar(r1, psd_all, width=barWidth, color=psd_color, edgecolor='white', label='All PSD')
bars2 = plt.bar(r2, psd_sel, width=barWidth, color=psd_color, edgecolor='white', hatch='///', alpha=0.7, label='Selected PSD')
bars3 = plt.bar(r3, csp_all, width=barWidth, color=csp_color, edgecolor='white', label='All CSP')
bars4 = plt.bar(r4, csp_sel, width=barWidth, color=csp_color, edgecolor='white', hatch='///', alpha=0.7, label='Selected CSP')

# Add a subtle divider between PSD and CSP for each classifier
for i in range(len(classifiers)):
    plt.axvline(x=r2[i] + barWidth/2, color='gray', linestyle='--', alpha=0.3)

# Group labels with larger font
for i in range(len(classifiers)):
    mid_point = (r1[i] + r4[i]) / 2
    plt.annotate(classifiers[i], xy=(mid_point, 0.75), xytext=(0, -30), 
                 textcoords="offset points", ha='center', fontweight='bold', fontsize=16)

# Add labels and title with larger font
plt.ylabel('Accuracy', fontweight='bold', fontsize=16)
plt.title('Accuracy Comparison Across Different Feature Extraction Methods', fontweight='bold', fontsize=18)

# Remove the classifiers from x-ticks as we've added annotations
plt.xticks([])

# Add value labels on top of bars with larger font
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.4f}', ha='center', va='bottom', fontsize=11, rotation=0)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

# Add a grid for better readability
plt.grid(True, axis='y', linestyle='--', alpha=0.5)

# Set y-axis limits for better visualization
plt.ylim(0.75, 0.95)

# Make y-tick labels larger
plt.yticks(fontsize=14)

# Create a more readable legend using custom elements with larger font
legend_elements = [
    Patch(facecolor=psd_color, edgecolor='white', label='All PSD'),
    Patch(facecolor=psd_color, edgecolor='white', hatch='///', alpha=0.7, label='Selected PSD'),
    Patch(facecolor=csp_color, edgecolor='white', label='All CSP'),
    Patch(facecolor=csp_color, edgecolor='white', hatch='///', alpha=0.7, label='Selected CSP')
]
plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.07), 
           ncol=4, frameon=True, fontsize=14)

# REMOVED: Text annotation for PSD and CSP
# The following two lines have been removed:
# plt.text(r1[0] + barWidth/2, 0.76, "PSD", ha='center', va='bottom', color=psd_color, fontweight='bold', fontsize=14)
# plt.text(r3[0] + barWidth/2, 0.76, "CSP", ha='center', va='bottom', color=csp_color, fontweight='bold', fontsize=14)

# Add tight layout
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# Save the figure
plt.savefig('accuracy_comparison_large_text.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()