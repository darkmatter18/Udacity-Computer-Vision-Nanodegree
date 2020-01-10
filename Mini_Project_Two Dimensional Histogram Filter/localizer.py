#import pdb
from helpers import normalize, blur

def initialize_beliefs(grid):
    height = len(grid)
    width = len(grid[0])
    area = height * width
    belief_per_cell = 1.0 / area
    beliefs = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append(belief_per_cell)
        beliefs.append(row)
    return beliefs

def sense(color, grid, beliefs, p_hit, p_miss):
    new_beliefs = []
    for i in range(len(beliefs)):
        x  =  []
        for j in range(len(beliefs[0])):
            if grid[i][j] == color:
                x.append(beliefs[i][j] * p_hit)
            else:
                x.append(beliefs[i][j] * p_miss)
        new_beliefs.append(x)
    new_beliefs = normalize(new_beliefs)
    #
    # TODO - implement this in part 2
    #

    return new_beliefs

def move(dy, dx, beliefs, blurring):
    height = len(beliefs)
    width = len(beliefs[0])
    new_G = [[0.0 for i in range(width)] for j in range(height)]
    for i, row in enumerate(beliefs):
        for j, cell in enumerate(row):
            new_i = (i + dy ) % width
            new_j = (j + dx ) % height
            #pdb.set_trace()
            new_G[int(new_j)][int(new_i)] = cell
    return blur(new_G, blurring)