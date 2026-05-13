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

accuracies = [
    80.66,
    82.22,
    90.00,
    0,
    99.91,
    85.70
]

plt.figure(figsize=(10,6))

plt.bar(models, accuracies)

plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Comparison of Model Accuracy')

plt.ylim(0, 100)

for i, value in enumerate(accuracies):

    if value == 0:
        plt.text(i, 2, 'WAITING', ha='center')
    else:
        plt.text(i, value + 1, f'{value:.2f}%', ha='center')

plt.tight_layout()

plt.savefig('accuracy_comparison.png', dpi=300)

print("Accuracy graph saved successfully")