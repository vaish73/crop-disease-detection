import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

models = [
    'CNN',
    'MobileNetV2',
    'ResNet50',
    'DenseNet121',
    'Vision Transformer',
    'SVM'
]

recall = [
    78,
    80,
    89,
    0,
    99,
    84
]

plt.figure(figsize=(10,6))

plt.bar(models, recall)

plt.xlabel('Models')
plt.ylabel('Recall (%)')
plt.title('Recall Comparison Across Models')

plt.ylim(0,100)

for i, value in enumerate(recall):

    if value == 0:
        plt.text(i, 2, 'WAITING', ha='center')
    else:
        plt.text(i, value + 1, f'{value}%', ha='center')

plt.tight_layout()

plt.savefig('recall_comparison.png', dpi=300)

print("Recall graph saved successfully")