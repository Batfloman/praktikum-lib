import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from batfloman_praktikum_lib.graph_fit.init_params._widget_helper import mult_slider_range, recenter_slider

fig, ax = plt.subplots()

slider = Slider(ax, valmin=0, valmax=10, valinit=5, label="")

def test(val):
    print("Test", val)

slider.on_changed(test)

# Example key press callback
def on_key(event):
    if event.key == "r":
        recenter_slider(slider)
        print("Recentered")
    if event.key == "e":
        mult_slider_range(slider, 2)
        print("Expand")
    if event.key == "t":
        mult_slider_range(slider, 0.5)
        print("Shrink")
    elif event.key == "q":  # Press 'q' to quit the figure
        plt.close(fig)

# Connect the key press
fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()
