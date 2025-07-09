import click
import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@click.group()
def cli():
    """Nepali Language Model Suite CLI."""
    pass


@cli.command()
def list():
    """List all available experiments."""
    from nep_lms.core.experiment import ExperimentRegistry

    exp_names = ExperimentRegistry.get_experiment_names()
    if exp_names:
        click.echo("Available experiments:")
        for name in exp_names:
            click.echo(f"- {name}")
    else:
        click.echo("No experiments found.")


@cli.command()
@click.argument("experiment_name")
def run(experiment_name: str):
    """Run an experiment."""
    from nep_lms.core.experiment import ExperimentRegistry

    ExperimentRegistry.run_experiment(experiment_name)


if __name__ == "__main__":  # pragma: no cover
    cli()
