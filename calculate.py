from ase import Atoms
from ase.build import bulk
from ase.io import read
from gpaw import GPAW, restart
from gpaw.wavefunctions.pw import PW
from gpaw.occupations import FermiDirac
import numpy as np
from pathlib import Path
from gpaw.mpi import world
from gpaw.response.bse import BSE
from gpaw.response.df import DielectricFunction
from ase.parallel import parprint
import datetime
from gpaw.external import ConstantElectricField
import matplotlib.pyplot as plt
from ase.dft.bandgap import bandgap as get_gap


def main():
    efield = 0
    fname = "mono_nano.cif"
    do_gs_calc(fname)
    for efield in np.linspace(0, 2, 5):
        add_efield(fname, efield)


def time():
    return '{0:%Y-%m-%d %H:%M:%S} '.format(datetime.datetime.now())


def printit(text, fname="output.text"):
    parprint(time()+text, file=open(fname, "a"))
    parprint(text)


def do_gs_calc(fname):
    printit('running GS calculation')
    a = read(fname)

    calc = GPAW(mode='lcao',
                basis='dzp',
                xc='PBE',
                occupations=FermiDirac(width=0.01),
                kpts={'size': (1, 20, 1), 'gamma': True},
                txt='gs_output.txt')

    a.set_calculator(calc)
    a.get_potential_energy()
    calc.write('gs_'+fname+'.gpw', mode='all')
    printit('GS calculation done')


def add_efield(fname, efield=0):
    printit('Starting Efield calculation for efiled = {}'.format(efield))
    atom, calc = restart('gs_'+fname+'.gpw')
    calc.set(txt='gs_efield_{}.txt'.format(efield))
    calc.set(external=ConstantElectricField(efield, [0, 0, 1]))
    calc.get_potential_energy()
    calc.write('gs_'+fname+'_{}.gpw'.format(efield), mode='all')
    printit('Efield GS calculation done')
    gap, p1, p2 =  get_gap(calc)
    printit("{} {} {} {}".format(efield, gap,p1,p2), fname="gaps.dat")
    printit("efield={} gap={}".format(efield, gap))
    # band_calc('gs_'+fname+'_{}.gpw'.format(efield))


def band_calc(fname):
    printit('Starting Band structure calculation')
    atom, calc = restart(fname)
    path = atom.cell.bandpath(
        [[0, 0, 0], [0, 0.5, 0], [0, 1, 0]], npoints=100)
    calc.set(fixdensity=True)
    calc.set(symmetry='off')
    calc.set(kpts=path.kpts)
    calc.get_potential_energy()
    printit('Band structure calculation Done')
    gap, p1, p2 = gap(calc)
    printit("{} {}".format(fname, gap), fname="gaps.dat")
    e_kn = np.array([calc.get_eigenvalues(k) for k in range(len(kpts))])
    efermi = calc.get_fermi_level()
    plt.figure(figsize=(5, 6))
    x = path.get_linear_kpoint_axis()[0]
    for i in range(nbands):
        plt.plot(x, e_kn[:, i])
    for j in path.get_linear_kpoint_axis()[1]:
        plt.plot([0, X[-1]], [0, 0], 'k-')
    plt.axis(xmin=0, xmax=X[-1], ymin=emin, ymax=emax)
    plt.tight_layout()
    plt.savefig(fname+'.png', dpi=300)


if __name__ == "__main__":
    main()
