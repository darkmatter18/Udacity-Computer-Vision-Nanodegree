import matplotlib.pyplot as plt
'''
The color class creates a color from 3 values, r, g, and b (red, green, and blue).

attributes:
    r - a value between 0-255 for red
    g - a value between 0-255 for green
    b - a value between 0-255 for blue
'''

class Color(object):

    # Initializes a color with rgb values
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    # Called when a Color object is printed out
    def __repr__(self):
        '''Display a color swatch and returns a text description of r,g,b values'''
        
        plt.imshow([[(self.r/255, self.g/255, self.b/255)]])
        
        return 'r, g, b = ' + str(self.r) + ', ' + str(self.g) + ', ' + str(self.b)
    

    ## TODO: Complete this add function to add two colors together
    def __add__(self, other):
        '''Adds the r, g, and b components of each color together 
           and averaging them. 
           The new Color object, with these averaged rgb values, 
           is returned.'''
        r = (self.r + other.r)/2
        g = (self.g + other.g)/2
        b = (self.b + other.b)/2
        
        return Color(r, g, b)



