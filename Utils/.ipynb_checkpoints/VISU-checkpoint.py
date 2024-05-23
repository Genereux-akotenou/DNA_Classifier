class VISU:
    @staticmethod
    def plot_curve(history, list_of_metrics):
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        epochs = history.epoch
        hist = pd.DataFrame(history.history)
        for m in list_of_metrics:
            x = hist[m]
            plt.plot(epochs[1:], x[1:], label=m, lw=2)
        plt.legend()