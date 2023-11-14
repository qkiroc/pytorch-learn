import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os

def plotLearningCurve(loss_record, title=""):
    total_steps = len(loss_record)
    x_1 = range(total_steps)
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record, c="tab:red", label="train")
    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.title("Learning curve of {}".format(title))
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'loss.png'))

