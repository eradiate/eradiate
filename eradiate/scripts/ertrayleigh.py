import click
import copy
import matplotlib.pyplot as plt
import yaml

from eradiate.solvers.onedim.rayleigh import RayleighSolverApp
import eradiate.util.view as view


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.argument("fname_results", type=click.Path())
@click.argument("fname_plots", type=click.Path())
@click.option("-q", "--quiet", is_flag=True, help="Hide progress information.")
def cli(config, fname_results, fname_plots, quiet):
    """A simple command-line interface to the RayleighSolverApp class.

    This tool reads a RayleighSolverApp YAML configuration file located at
    CONFIG and simulates radiative transfer on it. It stores the results in a file located
    at FNAME_RESULTS and adds a series of default views under FNAME_PLOTS.
    """

    # Load configuration
    with open(config, "r") as configfile:
        config = yaml.safe_load(configfile)
    app = RayleighSolverApp(config)

    # Run simulation
    app.run(fname_results=fname_results)

    if fname_plots:
        for result in app.results:
            for quantity, data in result.items():
                if quantity == "irradiance":
                    continue
                if data.attrs["angular_domain"] == "hsphere":
                    ax = plt.subplot(111, projection="polar")
                    data.ert.plot(kind="polar_pcolormesh", title=quantity, ax=ax)
                elif data.attrs["angular_domain"] == "pplane":
                    ax = plt.subplot(111)
                    plane = view.plane(data)
                    plane.ert.plot(kind="linear", title=quantity, ax=ax)
                plt.savefig(f"{fname_plots}_{quantity}.png", bbox_inches="tight")
                plt.close()


if __name__ == "__main__":
    cli()
