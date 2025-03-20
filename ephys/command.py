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
@click.option('--save_file', is_flag=True, default=True)
def probe(path, save_file):
    from .utils import finder
    from .spikeglx import plot_chanmap
    fn = finder(path, r'ap.meta$')
    plot_chanmap(fn, save_file=save_file)

@main.command()
@click.option('--path', type=click.Path(exists=True), default=None)
@click.option('--metric', is_flag=True)
@click.option('--ks2', is_flag=True)
def runks(path, metric, ks2):
    if ks2:
        from .ks import run_ks2, Kilosort
        fns = run_ks2(path)
        if metric:
            print('Not implemented for Kilosort2. Run `ephys metric` to calculate metrics.')
    else:
        from .ks import run_ks4, Kilosort
        fns = run_ks4(path)

        if metric:
            for fn in fns:
                ks_dir = os.path.join(os.path.dirname(fn), 'kilosort4')
                ks_fn = os.path.join(ks_dir, 'params.py')
                if os.path.exists(ks_fn):
                    ks = Kilosort(ks_dir)
                    ks.save_metrics()

@main.command()
@click.option('--path', type=click.Path(exists=True), default=None)
def metric(path):
    from .utils import finder
    from .ks import Kilosort
    fn = finder(path, 'params.py$')
    if fn is None:
        raise ValueError('No params.py found in the directory')
    fd = os.path.dirname(fn)
    ks = Kilosort(fd)
    ks.save_metrics()

@main.command()
@click.option('--path', type=click.Path(exists=True), default=None)
@click.option('--metric', is_flag=True)
@click.option('--bmi', is_flag=True)
@click.option('--all', is_flag=True)
@click.option('--mua', is_flag=True)
def saveks(path, metric, bmi, all, mua):
    from .utils import finder
    from .ks import Kilosort
    
    fn = finder(path, 'params.py$')
    if not fn:
        raise ValueError('No params.py found in the directory')
    
    fd = os.path.dirname(fn)
    group = 'all' if all else 'mua' if mua else 'good'
    
    ks = Kilosort(fd, group=group)
    ks.load_waveforms()
    
    if not bmi:
        ks.load_sync()
        ks.load_nidq()
        ks.load_obx()
    
    if metric:
        ks.save_metrics()
    
    ks.save()

@main.command()
@click.option('--path', type=click.Path(exists=True), default=None)
@click.option('--nidq', is_flag=True)
@click.option('--tdms', is_flag=True)
def bmi(path, nidq, tdms):
    from .bmi import BMI
    bmi = BMI(path)
    bmi.save_mua()
    if nidq:
        bmi.save_nidq()
    elif tdms:
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