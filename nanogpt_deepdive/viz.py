import io
from PIL import Image
import matplotlib.pyplot as plt


def plot_histogram(data, bins=10, xlabel="", ylabel="", title=""):
    """
    Plot a histogram of the given data.

    Parameters:
    - data: A list or array-like object containing the data points.
    - bins: Number of bins in the histogram. (default = 10)
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - title: Title of the histogram plot.
    """
    plt.hist(data, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    image = Image.open(buf)
    plt.close()
    return image
