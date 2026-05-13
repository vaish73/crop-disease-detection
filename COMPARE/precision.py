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

precision = [
    79,
    81,
    90,
    0,
    99,
    83
]

plt.figure(figsize=(10,6))

plt.bar(models, precision)

plt.xlabel('Models')
plt.ylabel('Precision (%)')
plt.title('Precision Comparison Across Models')

plt.ylim(0,100)

for i, value in enumerate(precision):

    if value == 0:
        plt.text(i, 2, 'WAITING', ha='center')
    else:
        plt.text(i, value + 1, f'{value}%', ha='center')

plt.tight_layout()

plt.savefig('precision_comparison.png', dpi=300)

print("Precision graph saved successfully")