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
    from .ks import Kilosort
    fn = finder(path, 'params.py$')
    if fn is None:
        raise ValueError('No params.py found in the directory')
    fd = os.path.dirname(fn)
    ks = Kilosort(fd)
    ks.save()

@main.command()
@click.option('--path', type=click.Path(exists=True), default=None)
def savebmi(path):
    from .bmi import BMI
    bmi = BMI(path)
    bmi.save_mua()
    bmi.save_nidq()

@main.command()
@click.option('--path', type=click.Path(exists=True), default=None)
def savetdms(path):
    from .tdms import save_tdms
    save_tdms(path)

@main.command()
@click.option('--path', type=click.Path(exists=True), default=None)
def savevr(path):
    from .vr import VR
    vr = VR(path=path)
    vr.save()