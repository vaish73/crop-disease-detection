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

accuracy = [80.66, 82.22, 90.00, 0, 99.91, 85.70]
precision = [79, 81, 90, 0, 99, 83]
recall = [78, 80, 89, 0, 99, 84]
f1 = [79, 81, 89, 0, 99, 84]

plt.figure(figsize=(12,7))

plt.plot(models, accuracy, marker='o', label='Accuracy')
plt.plot(models, precision, marker='o', label='Precision')
plt.plot(models, recall, marker='o', label='Recall')
plt.plot(models, f1, marker='o', label='F1-Score')

plt.xlabel('Models')
plt.ylabel('Percentage (%)')

plt.title('Performance Comparison of Different Models')

plt.ylim(0,100)

plt.legend()

plt.grid(True)

# Add WAITING text for DenseNet121
dense_index = models.index('DenseNet121')

plt.text(dense_index, 2, 'WAITING', ha='center')

plt.tight_layout()

plt.savefig('overall_model_comparison.png', dpi=300)

print("Overall comparison graph saved successfully")