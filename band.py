from ase import Atoms
from ase.build import bulk
from ase.io import read
from gpaw import GPAW, restart, Davidson, Mixer, PoissonSolver
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
import os.path
from ase.build import make_supercell


def time():
    return '{0:%Y-%m-%d %H:%M:%S} '.format(datetime.datetime.now())


def printit(text, fname="output.text"):
    parprint(time()+text, file=open(fname, "a"))
    parprint(text)


def efield(efield=0.05, n=30):
    printit('running GS calculation for {}'.format(efield))
    fname = "sb_relaxed.cif"
    a = read(fname)
    atom = read(fname)
    wire = make_supercell(atom, [[n, 0, 0], [0, 1, 0], [
                          0, 0, 1]], wrap=True, tol=1e-05)
    wire.center(vacuum=15, axis=0)
    a = wire.copy()
    if not os.path.isfile('gs_{}.gpw'.format(efield)):
        calc = GPAW(  # mode='lcao',
            mode='fd',
            basis='szp(dzp)',
            # eigensolver='cg',
            #mixer=Mixer(beta=0.2, nmaxold=3, weight=70.0),
            xc='PBE',
            # poissonsolver=DipoleCorrection(PoissonSolver(relax='GS'), 2),
            mixer=Mixer(beta=0.06, nmaxold=5, weight=100.0),
            occupations=FermiDirac(width=0.1),
            kpts={'density': 4.5},
            txt='gs_output_{}.txt'.format(efield),
            h=0.20,
            external=ConstantElectricField(efield, [0, 0, 1]))

        a.set_calculator(calc)
        a.get_potential_energy()
        calc.write('gs_{}.gpw'.format(efield))

    calc = GPAW('gs_{}.gpw'.format(efield),
                nbands=-100,
                fixdensity=True,
                symmetry='off',
                kpts={'path': 'XGX', 'npoints': 100},
                convergence={'bands': 'CBM+1'},
                txt='gs_output_{}.txt'.format(efield))
    calc.get_potential_energy()
    bs = calc.band_structure()
    bs.plot(filename='bandstructure_{}.png'.format(
        efield), show=False, emax=calc.get_fermi_level()+1, emin=calc.get_fermi_level()-1)
    bs.write('bs_{}.json'.format(
        efield))
    try:
        gap, p1, p2 = get_gap(calc)
        printit("{} {} {} {}".format(efield, gap, p1, p2), fname="gaps.dat")
    except:
        None


efield(efield=2, n=25)
