import os
import click

@click.group()
def main():
    pass

@main.command()
@click.option('--path', type=click.Path(exists=True), default=None)
def catgt(path):
    from ephys.catgt import run_catgt
    run_catgt(path)

@main.command()
@click.option('--path', type=click.Path(exists=True), default=None)
def runks(path):
    from ephys.kilosort import run_kilosort
    run_kilosort(path)

@main.command()
@click.option('--path', type=click.Path(exists=True), default=None)
def saveks(path):
    from ephys.util import finder
    from ephys.kilosort import Spike
    fn = finder(path, 'params.py$')
    if fn is None:
        raise ValueError('No params.py found in the directory')
    fd = os.path.dirname(fn)
    spike = Spike(fd)
    spike.save()

@main.command()
@click.option('--path', type=click.Path(exists=True), default=None)
def spiketag(path):
    pass

