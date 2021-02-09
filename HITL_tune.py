"""
tune the parameters of the model in silico to visually match invitro data using
gaussian process based human-in-the-loop optimization
"""

from cell import *
from animate import *
import tkinter as tk
import threading
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


score = 0  #global variable that objective fxn manipulates
def HITL_object_fun(x):
    """
    wrapper function which runs a CellMech simulation with the supplied
    parameters, and then asks the user to score the extent to which the simulation
    reflects the invivo results for a particular cell line. 
    
    inputs: x is a vector containing
        d0max  - max distance connected by links
        p_add  - rate to add cell-cell links
        p_del  - rate to delete cell-cell links
    returns: 
        d
    """
    
    #------------------------ setup the constants -----------------------------
    # run simulation of cells initialized in square above substrate
    d0max,p_add,p_del = x
    
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
    
    #ask for a grade from the expert
    def close_window():
        global score
        score = int(e.get())
        root.destroy()
        
    def enter_callback(event):
        close_window()
    
    root = tk.Tk()
    tk.Label(root, text="enter score for this simulation on [0-100], then press enter or submit").grid(row=0)
    e = tk.Entry(root)
    e.grid(row=1, column=0)   
    root.bind("<Return>", enter_callback)  
    b = tk.Button(root, text = "submit score", command = close_window)
    b.grid(row=1, column=1)  
    root.mainloop()
    
    #close the mlab visualization and return the score
    #mlab.close()
    return score 

    
    
    
    
#------------------------ run the HILT Optimization ---------------------------

if __name__ == "__main__":
    
    from skopt import Optimizer
    
    #define boundary condition limits (lower,upper)
    d0max_lims = (1.5,2.5)
    p_add_lims = (.8,1.2)
    p_del_lims = (.15,.25)
    bounds = [d0max_lims,p_add_lims,p_del_lims]
    
    nSamps = 3
    opt = Optimizer(bounds)
    for i in range(nSamps):
        suggested = opt.ask()
        y = HITL_object_fun(suggested)
        res = opt.tell(suggested, y)
    
    



#---------------------- notes for future improvements ------------------------

# can we get rid of the need to save stuff to disk without breaking things, this will speed up calculations.
# getting the problem with mlab - even when we don't have tk code there..... why? 
