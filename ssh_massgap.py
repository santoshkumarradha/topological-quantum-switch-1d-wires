import numpy as np
import matplotlib.pyplot as plt
import copy
import sb_model as t
import pybinding as pb
import matplotlib.gridspec as gridspec
import sys
p = [
    '/Users/santy/Desktop/Physics Projects',
    '/usr/local/Caskroom/miniconda/base/lib/python37.zip',
    '/usr/local/Caskroom/miniconda/base/lib/python3.7',
    '/usr/local/Caskroom/miniconda/base/lib/python3.7/lib-dynload',
    '/usr/local/Caskroom/miniconda/base/lib/python3.7/site-packages'
]
for i in p:
    sys.path.append(i)
from matplotlib import cm
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


def model_ssh(t1=1, t2=1, m=0, n=26, m1=0):
    """returns SSH model

    Args:
        t1 (int, optional): incell interaction
        t2 (int, optional): inter cell interaction
        m (int, optional): mass gap
        n (int, optional): number of cells
        m1 (int, optional): alternate mass gap difference

    Returns:
        TYPE: Description
    """
    def mass_term(delta):
        """Break sublattice symmetry with opposite A and B onsite energy"""
        @pb.onsite_energy_modifier
        def potential(energy, x, y, z, sub_id):
            pot = []
            for i in range(len(x)):
                if x[i] < 0:
                    if sub_id in ['A']:
                        pot.append(-delta)
                    if sub_id in ['B']:
                        pot.append(delta)
                else:  #if ang<=0 and ang>-180:
                    if sub_id in ['A']:
                        pot.append(delta)
                    if sub_id in ['B']:
                        pot.append(-delta)
            return np.array(pot)

        return potential

    d = 1
    t1 = t1
    t2 = t2
    m = m
    m1 = m1
    lattice = pb.Lattice(a1=[d])
    lattice.add_sublattices(('A', [0], 0), ('B', [.5], 0))
    lattice.add_hoppings(([0, 0], 'A', 'B', t1), ([1, 0], 'B', 'A', t2))
    model = pb.Model(lattice, pb.rectangle(n), mass_term(m),
                     pb.translational_symmetry(a1=False))
    return model


def plot_sshwav():
    """Plot SSH wavefunctions
	"""
    fig, ax = plt.subplots(figsize=(5.5, 2))
    e1 = int(len(e) / 2) * 0
    probability_map = solver.calc_probability([e1])
    ax.scatter(probability_map.x,
               probability_map.y,
               s=probability_map.data * 1500,
               c="r",
               alpha=.5,
               zorder=2)
    ax.scatter(probability_map.x,
               probability_map.y,
               s=5,
               c="k",
               alpha=.5,
               zorder=1)
    ax.axis('off')
    plt.savefig("sshmass_plot.png", dpi=600)
    plt.show()


def plot_soliton_massgap(n=7):
    """plot soliton mass energy spectrum

	Args:
	    n (int, optional): number of system
	"""
    E = []
    pos = []
    for i in np.linspace(-2, 2, n):
        t1 = 1.5
        t2 = 1
        m = i
        model = model_ssh(t1=t1, t2=t2, m=m, m1=0, n=30)
        solver = pb.solver.lapack(model)
        e = solver.calc_eigenvalues(map_probability_at=[0, 0])
        E.append(e)
        e2 = solver.calc_eigenvalues(map_probability_at=[model.system.x.max()])
        pos.append(e2.probability.argsort()[-2:][::-1])
    fig = plt.figure(figsize=(6, 3), constrained_layout=True)
    plt.style.use("science")
    spec2 = gridspec.GridSpec(ncols=n, nrows=1, figure=fig)

    for iax, e in enumerate(E):
        #norms=e.probability / np.linalg.norm(e.probability)
        norms = e.probability / np.max(e.probability)
        colors = [cm.autumn_r(x) for x in norms]
        ax = fig.add_subplot(spec2[0, iax])
        mask = np.ones(len(e.values), np.bool)
        mask[pos[iax]] = 0
        e1 = e.values[mask]
        for j, i in enumerate(e.values):
            if j not in pos[iax]:
                ax.plot([0, 1], [i, i], c=colors[j], lw=1.5, zorder=1)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticklabels("")
                ax.xaxis.set_ticks_position('none')
                ax.scatter(0.5,
                           e1[int(len(e1) / 2 - 1)],
                           edgecolor="k",
                           zorder=2,
                           facecolor="none")
    plt.savefig("sshmass.png", dpi=600)
    plt.show()
