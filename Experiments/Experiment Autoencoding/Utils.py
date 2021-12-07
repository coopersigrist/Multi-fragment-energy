import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")


# visualizes progress over training

def plot_epochs_history(data, title, x_axis, y_axis):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()
