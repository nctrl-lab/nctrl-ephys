import os
import click

@click.group()
def main():
    pass

@main.command()
@click.option('--path', type=click.Path(exists=True), default=None)
def catgt(path):
    from .catgt import run_catgt
    run_catgt(path)

@main.command()
@click.option('--path', type=click.Path(exists=True), default=None)
def runks(path):
    from .ks import run_ks4
    run_ks4(path)

@main.command()
@click.option('--path', type=click.Path(exists=True), default=None)
def saveks(path):
    from .utils import finder
    from .ks import Spike
    fn = finder(path, 'params.py$')
    if fn is None:
        raise ValueError('No params.py found in the directory')
    fd = os.path.dirname(fn)
    spike = Spike(fd)
    spike.save()

@main.command()
@click.option('--path', type=click.Path(exists=True), default=None)
def savebmi(path):
    from .bmi import BMI
    bmi = BMI(path)
    bmi.save_mua()
    bmi.save_nidq()