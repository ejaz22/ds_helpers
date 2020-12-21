import keras
import matplotlib.pyplot as plt


def plot_history(history: keras.callbacks.History):
    metrics = [metric for metric in history.history.keys() if not metric.startswith('val_')]
    stride = len(history.epoch)//20
    plotted_epochs = history.epoch[::stride]
    
    fig, subplots = plt.subplots(len(metrics), figsize=(8, 4*len(metrics)))
    subplots = subplots if len(metrics) != 1 else (subplots,)
    fig.tight_layout(h_pad=3, rect=[0, 0, 1, 0.95])
    fig.suptitle('Model training history', fontsize=18)
    
    for metric, subplot in zip(metrics, subplots):
        subplot.plot(plotted_epochs, history.history[metric][::stride], marker='.')
        try: subplot.plot(plotted_epochs, history.history[f'val_{metric}'], marker='.')
        except KeyError: pass
        subplot.set_xticks(plotted_epochs)
        subplot.set_ylabel(metric)
        subplot.set_xlabel('epoch')
    
    if len(metrics) != len(history.history):
        fig.legend(['training', 'validation'])
