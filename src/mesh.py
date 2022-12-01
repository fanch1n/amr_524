#==============================================================================
# classes 
#==============================================================================
# Generating adaptive meshes for a field 

"""
define the class of problem called "phases"
M -> 2D field matrix, containing information such as phases and  concentration
maxRes -> sets the number of mesh levels that the domain is allowed to have
threshold -> sets the level of refinement
"""

class Mesh:
    # initialize the class by the matrix itself, gradient matrix
    def __init__(self, M, maxRes, gradient):
        # matrix
        self.M = M            
        # max number of refinement levels
        self.maxRes = maxRes
        # gradient matrix
        self.gradient = gradient
        
    def criterion(self, EPS):
        return 0