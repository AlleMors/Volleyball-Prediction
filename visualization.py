import json
import numpy as np
from matplotlib import pyplot as plt

def print_feature_importance(model, feature_names, output_file='feature_importance.json'):
    # Extract and visualize feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 8))
    plt.title("Feature Importance")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)),
               [f"{feature_names[i]} ({importances[i]:.4f})" for i in indices],
               rotation=90)
    plt.tight_layout()
    # plt.savefig("feature_importance.png")

    # Create a dictionary with feature names and their importances
    feature_importance_dict = {feature_names[i]: importances[i] for i in indices}

    # Save JSON to file
    with open(output_file, 'w') as f:
        json.dump(feature_importance_dict, f, indent=4)

    plt.show()