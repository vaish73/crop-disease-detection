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

f1_scores = [
    79,
    81,
    89,
    0,
    99,
    84
]

plt.figure(figsize=(10,6))

plt.bar(models, f1_scores)

plt.xlabel('Models')
plt.ylabel('F1-Score (%)')
plt.title('F1-Score Comparison Across Models')

plt.ylim(0,100)

for i, value in enumerate(f1_scores):

    if value == 0:
        plt.text(i, 2, 'WAITING', ha='center')
    else:
        plt.text(i, value + 1, f'{value}%', ha='center')

plt.tight_layout()

plt.savefig('f1score_comparison.png', dpi=300)

print("F1-Score graph saved successfully")