import click
import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@click.group()
def cli():
    """Nepali Language Model Suite CLI."""
    pass


if __name__ == "__main__":  # pragma: no cover
    cli()
