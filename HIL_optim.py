"""
tune the parameters of the model in silico to visually match invitro data using
gaussian process based human-in-the-loop optimization
"""

import tkinter as tk
import os
from skopt import Optimizer


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
        score
    """
    
    #extract and run CellMech in a separate kernal
    d0max,p_add,p_del = x
    os.system("python HIL_objective.py " +
              str(d0max) + " " + 
              str(p_add) + " " +
              str(p_del))
   
    #callback functions
    def close_window():
        global score
        score = int(e.get())
        root.destroy()
        
    def enter_callback(event):
        close_window()
    
    #ask for a grade from the expert, block until done
    root = tk.Tk()
    tk.Label(root, text="enter score for this simulation on [0-100], then press enter or submit").grid(row=0)
    e = tk.Entry(root)
    e.grid(row=1, column=0)   
    root.bind("<Return>", enter_callback)  
    b = tk.Button(root, text = "submit score", command = close_window)
    b.grid(row=1, column=1)  
    root.mainloop()
    
    return score 

    
    
    
#------------------------ run the HILT Optimization ---------------------------

if __name__ == "__main__":
    
    """
    instructions: 
        - hardcode the bounds of the parameters you want to explore
        - hardcode the number of samples you want to collect
        - run file
    """
    
    #define boundary condition limits (lower,upper)
    d0max_lims = (1.5,2.5)
    p_add_lims = (.8,1.2)
    p_del_lims = (.15,.25)
    bounds = [d0max_lims,p_add_lims,p_del_lims]
    
    nSamps = 5
    opt = Optimizer(bounds)
    for i in range(nSamps):
        suggested = opt.ask()
        y = HITL_object_fun(suggested)
        res = opt.tell(suggested, y)
    
    print(res)
    
    #plots
    from skopt.plots import plot_objective, plot_evaluations
    plot_evaluations(res, bins=10)
    plot_objective(res)
    



#---------------------- notes for future improvements ------------------------

# can we get rid of the need to save stuff to disk without breaking things, this will speed up calculations.
# getting the problem with mlab - even when we don't have tk code there..... why? 
# add code to plot the results.
# is there a way to save a model, so that you dont' have to start from scratch each time?
