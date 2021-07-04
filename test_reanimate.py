# this file takes the files cached in a results directory and animates them
from cell import *
from animate import *

#where did you save the results? 
savedir = "res"            


#re-initialize tct from the directory   
tct = TwoCelltypePatch(10)
tct.load(savedir)


#load the simulation results from disk
dumpdata, simdata, subssimdata = fetchdata(savedir)

# animate results
animateconfigs(simdata, subssimdata, showsubs=False,cell_types = tct,fiji=True)
mlab.show()
