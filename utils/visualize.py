import PIL
import numpy as np
import matplotlib.pyplot as plt
def create_scatter_plot(x, y, title, x_label, y_label):
    """
    Creates a scatter plot with the given x and y data, title, and axis labels.

    Parameters:
    x (array-like): The data for the x-axis.
    y (array-like): The data for the y-axis.
    title (str): The title of the plot.
    x_label (str): The label for the x-axis.
    y_label (str): The label for the y-axis.
    """
    if len(x) != len(y):
        raise ValueError("The length of x and y arrays must be the same.")

    sizes = np.arange(1, len(x) + 1)  # Generate size array for the markers

    plt.figure(figsize=(8, 6))  # Set the figure size
    scatter = plt.scatter(x, y, s=50, c=sizes, cmap='viridis')  # Create scatter plot with a colormap
    plt.title(title)  # Set the title
    plt.xlabel(x_label)  # Set x-axis label
    plt.ylabel(y_label)  # Set y-axis label
    plt.grid(True)  # Show grid
    for i, txt in enumerate(sizes):
        plt.annotate(txt, (x[i], y[i]), fontsize=12)  # Annotate each point with its corresponding size value

    fig = plt.gcf()
    fig.canvas.draw()
    pil_img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close()
    return pil_img