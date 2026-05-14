from matplotlib import pyplot as plt
from matplotlib._pylab_helpers import Gcf

from batfloman_praktikum_lib.graph.quickMethods import create_plot, show


def test_show_reregisters_closed_plot(monkeypatch):
    plt.close("all")
    plot = create_plot()
    fig, _ = plot

    plt.close(fig)
    assert Gcf.get_fig_manager(fig.number) is None

    shown = []
    monkeypatch.setattr(
        plt,
        "show",
        lambda block=True: shown.append((block, tuple(sorted(plt.get_fignums())))),
    )

    show(plot, ignore_quiet=True)

    assert Gcf.get_fig_manager(fig.number) is not None
    assert shown == [(True, (fig.number,))]


def test_show_accepts_iterables_of_plots(monkeypatch):
    plt.close("all")
    plot1 = create_plot()
    plot2 = create_plot()
    fig1, _ = plot1
    fig2, _ = plot2

    plt.close(fig1)
    plt.close(fig2)

    shown = []
    monkeypatch.setattr(
        plt,
        "show",
        lambda block=True: shown.append((block, tuple(sorted(plt.get_fignums())))),
    )

    show([plot1, plot2], ignore_quiet=True)

    assert shown == [(True, (fig1.number, fig2.number))]


def test_show_only_shows_requested_plots(monkeypatch):
    plt.close("all")
    plot1 = create_plot()
    plot2 = create_plot()
    plot3 = create_plot()
    fig1, _ = plot1
    fig2, _ = plot2
    fig3, _ = plot3

    shown = []
    monkeypatch.setattr(
        plt,
        "show",
        lambda block=True: shown.append((block, tuple(sorted(plt.get_fignums())))),
    )

    show([plot1, plot2], ignore_quiet=True)

    assert shown == [(True, (fig1.number, fig2.number))]
    assert Gcf.get_fig_manager(fig3.number) is not None


def test_show_without_arguments_delegates_to_matplotlib(monkeypatch):
    plt.close("all")
    plot1 = create_plot()
    plot2 = create_plot()
    fig1, _ = plot1
    fig2, _ = plot2

    shown = []
    monkeypatch.setattr(
        plt,
        "show",
        lambda block=True: shown.append((block, tuple(sorted(plt.get_fignums())))),
    )

    show(ignore_quiet=True)

    assert shown == [(True, (fig1.number, fig2.number))]


def test_show_without_arguments_reregisters_tracked_closed_plots(monkeypatch):
    plt.close("all")
    plot = create_plot()
    fig, _ = plot

    plt.close(fig)
    assert Gcf.get_fig_manager(fig.number) is None

    shown = []
    monkeypatch.setattr(
        plt,
        "show",
        lambda block=True: shown.append((block, tuple(sorted(plt.get_fignums())))),
    )

    show(ignore_quiet=True)

    assert Gcf.get_fig_manager(fig.number) is not None
    assert shown == [(True, (fig.number,))]
