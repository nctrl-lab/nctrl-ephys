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
@click.option('--metric', is_flag=True)
@click.option('--bmi', is_flag=True)
def saveks(path, metric, bmi):
    from .utils import finder
    from .ks import Kilosort
    fn = finder(path, 'params.py$')
    if fn is None:
        raise ValueError('No params.py found in the directory')
    fd = os.path.dirname(fn)
    ks = Kilosort(fd)
    ks.load_waveforms()
    if not bmi:
        ks.load_sync()
        ks.load_nidq()
    if metric:
        ks.load_waveforms_full()
        ks.load_metrics()
    ks.save()

@main.command()
@click.option('--path', type=click.Path(exists=True), default=None)
@click.option('--nidq', is_flag=True)
def bmi(path, nidq):
    from .bmi import BMI
    bmi = BMI(path)
    bmi.save_mua()
    if nidq:
        bmi.save_nidq()
    else:
        bmi.save_tdms()

@main.command()
@click.option('--path', type=click.Path(exists=True), default=None)
@click.option('--type', type=click.Choice(['unity', 'pi']), default='unity')
def task(path, type):
    from .task import Task
    task = Task(path=path, task_type=type)
    task.save()
    task.summary()
    task.plot()