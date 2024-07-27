import matplotlib.pyplot as plt
import pandas as pd

def scatter_features(X, y):
    num_features = X.shape[1]
    cols = 2  # number of columns for subplots
    rows = (num_features + 1) // cols  # number of rows for subplots    
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 6))
    axes = axes.flatten()  # flatten the axes array for easy iteration
    for i in range(num_features):
        feature = X.iloc[:, i]  # assuming X is a pandas DataFrame
        axes[i].scatter(feature[y == 0], np.zeros_like(feature[y == 0]), label='Class 0', marker='o')
        axes[i].scatter(feature[y == 1], np.ones_like(feature[y == 1]), label='Class 1', marker='x')
        axes[i].set_xlabel(f'Feature {i}')
        axes[i].set_ylabel('Class')
        axes[i].set_title(f'Scatter plot of Feature {i} vs Class')
        axes[i].legend()
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()


class VISU:
    @staticmethod
    def plot_curve_old(history, list_of_metrics):
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        epochs = history.epoch
        hist = pd.DataFrame(history.history)
        for m in list_of_metrics:
            x = hist[m]
            plt.plot(epochs[1:], x[1:], label=m, lw=2)
        plt.legend()

    @staticmethod
    def plot_curve(history, list_of_metrics):
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        epochs = history.epoch
        hist = pd.DataFrame(history.history)
        
        for m in list_of_metrics:
            x = hist[m]
            # Scale loss and val_loss to [0, 1] if values are greater than 1
            if m in ['loss', 'val_loss']:
                if x.max() > 1:
                    x = x / x.max()  # Scale to [0, 1]
            plt.plot(epochs[1:], x[1:], label=m, lw=2)
            
        plt.legend()