"""
HIL_objective.py - provides the objective function for the human-in-the-loop based
parameter tuining algorithm for the CellMech project. The reason that this code
is contained within it's own script, rather than a function inside HIL_optim is 
that the mayavi visualizer intermittantly freezes when it's not run in it's own
thread, as the top level function. 
"""

from cell import *
from animate import *
import sys
npr.seed(seed=0)



def generatePoint(L):
    """
    Produce random 3-dimensional coordinate on x-y-plane confined to square
    :param L: float, length of side of confining square
    :return: numpy array of shape (3,)
    """
    X0 = (npr.rand() - .5) * L
    Y0 = (npr.rand() - .5) * L
    Z0 = 0.
    return np.array([X0, Y0, Z0])


#-------------------------- extract cmd line args -----------------------------


#change this if opt explores different parameters
                             #0th index is name of file
d0max = float(sys.argv[1])   #max distance connected by links
p_add = float(sys.argv[2])   #rate to add cell-cell links
p_del = float(sys.argv[3])   #rate to delete cell-cell links



#----------------------------- objective function -----------------------------

# run simulation of cells initialized in square above substrate

#constants
Lmax = 5                    # Length of confining square for tissue cells
Lsubs = 5                   # Length of confining square for substrate cells
N = None                    # Number of cells. If None: int(2 * Lmax**2)
Nsubs = None                # Number of substrate cells. If None: int(Lmax**2)
runtime = 100.              # Length of simulation run
dims = 3                    # Number of dimensions for the given problem

dtrec = 0.                  # Periodicity of making configuration snapshots (done after every plasticity step if 0)
savedata = True             # Whether to write simulation results to file
savedir = "res"             # Directory to save the simulation results
dtsave = 10.                # Periodicity of writing data to hard drive (done only after end of runtime if None)

dt = 0.01                   # fundamental time unit, relevant only in combination with nmax
nmax = 1000                 # dt * nmax is the maximum time for mechanical equilibration
qmin = 0.001                # Threshhold tension beneath which the system is in mechanical equilibrium

d0min = 0.8                 # min distance between cells when initialized
#d0max = 2.                  # max distance connected by links
d0_0 = 1.                   # equilibrium distance of links (fundamental scaling of space)
#p_add = 1.                  # rate to add cell-cell links
#p_del = 0.2                 # rate to delete cell-cell links


if N is None:
    N = int(Lmax ** 2)
if Nsubs is None:
    Nsubs = int(Lsubs ** 2)

config = CellMech(N, num_subs=Nsubs, dt=dt, nmax=nmax, qmin=qmin, d0_0=d0_0, p_add=p_add, p_del=p_del,
                  d0max=d0max, dims=dims, issubs=True)


# initialize random positions for tissue cells in square
for ni in range(N):
    while True:
        R1 = generatePoint(Lmax)
        OK = True
        for nj in range(ni):
            d = np.linalg.norm(config.mynodes.nodesX[nj] - R1)
            if d < d0min:
                OK = False
                break
        if OK:
            break
    config.mynodes.nodesX[ni] = R1


# initialize random positions of substrate tissue
for ni in range(Nsubs):
    while True:
        R1 = generatePoint(Lsubs)
        R1[2] = -d0_0
        OK = True
        for nj in range(ni):
            d = np.linalg.norm(config.mysubs.nodesX[nj] - R1)
            if d < d0min:
                OK = False
                break
        if OK:
            break
    config.mysubs.nodesX[ni] = R1

# add links between cells adjacent in Voronoi tesselation and closer than d0min
allnodes = np.concatenate((config.mynodes.nodesX, config.mysubs.nodesX), axis=0)

for i, j in VoronoiNeighbors(allnodes, vodims=3):
    if np.linalg.norm(allnodes[i] - allnodes[j]) <= d0max:
        if (i < config.N) and (j < config.N):
            config.mynodes.addlink(i, j)
        elif (i >= config.N) and (j >= config.N):
            continue
        else:
            config.mysubs.addlink(i, j - config.N, config.mynodes.nodesX[i], config.mynodes.nodesPhi[i])

# run and save simulation
config.timeevo(runtime, dtrec=dtrec, savedata=savedata, savedir=savedir, dtsave=dtsave)
dumpdata, simdata, subssimdata = fetchdata(savedir)

# animate results
animateconfigs(simdata, subssimdata, showsubs=False)
mlab.show()

    
    