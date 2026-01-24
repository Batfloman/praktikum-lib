from matplotlib.widgets import Slider

def recenter_slider(slider: Slider):
    center = (slider.valmax + slider.valmin) / 2
    delta = slider.val - center

    slider.valmin += delta
    slider.valmax += delta
    # redraw
    slider.ax.set_xlim(slider.valmin, slider.valmax)
    slider.ax.figure.canvas.draw_idle()

def mult_slider_range(slider, factor):
    old_range = abs(slider.valmin - slider.valmax)
    new_range = factor * old_range

    delta = new_range / 2
    center = (slider.valmin + slider.valmax) / 2 

    slider.valmin = center - delta
    slider.valmax = center + delta
    # redraw
    slider.ax.set_xlim(slider.valmin, slider.valmax)
    slider.ax.figure.canvas.draw_idle()
