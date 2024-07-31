# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:41:03 2022

@author: dp

This module (ie. collection of related functions and variables) is a toolbox
of helpful functions for my Python codes
"""

import numpy as np
import math
from scipy.spatial import distance

def Poly2(a,b,c,x):
    ans = a*x*x + b*x + c
    return ans

def getIndex(value,res,indexMax,start_value = 0):
    """
    This function computes the index of a value based on a serie of values equally
    spaced by resoluvotion = res
    
    start_value + [0 - res[ ==> index = 0
    start_value + [res - 2 x res[ ==> index = 1 
    ...
    start_value + [(indexMax-1) x res - indexMax x res] ==> index = indexMax - 1
                     
    Parameters
    ----------
    value : float
    res : float
    indexMax : integer
        DESCRIPTION.

    Returns
    -------
    index : integer
    iWarning: 
    
    Utilization:
    ------------    
    This function is useful when looking for the index of a value inside a mapping
    form RayXpert:
    - dim = indexMax * res is the dimension of the mapping
    - start_value corresponds to the extremity of the mapping 
    - res = dim / # voxels
        
        def indexForX(x,translation,dim,res):
        #   This function return the index corresponding to the value x in the mapping
        #   The mapping is defined by:
        #       - translation : its origin (= pos - 0.5*dim)
        #       - dim: its dimension
        #       - res: its resolution (i.e. the number of voxel)
        #   if x in [translation ; translation + dim/res[ ==> index = 0
        #   if x in [...[ ==> index = 1
        #   if x in [translation + dim - dim/res ; translation + dim] ==> index = res

            index = int((x - translation)*res/dim)
            if (x == translation + dim): index = res # at the extremity of the mapping, index = imax
            iWarning = 0
            if x < translation or x > translation + dim: 
                iWarning = 1
                print('Warning: X is outside the mapping boundaries!', x)
            return index, iWarning
                        
    """
    iWarning = 0
    if res !=0:
        index = int((value-start_value)/res)
        dim = res*indexMax
        if (value == start_value + dim): index = indexMax-1 # at the extremity of the mapping, index = imax
        
        if value < start_value or value > start_value + dim: 
        #if index >= indexMax:
        #    index = indexMax-1
        #    print("WARNING: Maximum index (",indexMax,") was reached for value =",value)
            print('Warning: The value is outside the interval!', value)
            iWarning = 1
    else:
        print('ERROR: res = 0!')
        index = 0
    return index,iWarning

def find_nearest(array, value):
    """
    Note that a similar function exists in Numpy! Here it is:
    import numpy as np
    def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx] 




def DepthZero(theta,phi,dim):
#    theta1 = np.atan(dim[1]/dim[2])
    d0=1
#    if
    return d0

def cart2spher(x,y,z,option = 0):
    """
    This function transforms Cartesian coordinates into spherical coordinates
                     
    Parameters
    ----------
     - X
     - Y
     - Z
    """
    if z==0: phi = 90.
    elif (z<0): print('ERROR: z shoudl be positive, otherwise revisit the code')
    else: phi = math.atan(math.sqrt(x**2+y**2)/z)*180/math.pi
                            
    if x==0:
        if y > 0 : theta = 90. # 
        elif y < 0: theta = 270
        else: theta = 0
    elif x<0: theta = 180 + math.atan(y/x)*180/math.pi #Q2 & Q3
    elif x>0 and y>0: theta = math.atan(y/x)*180/math.pi #Q1
    else: 
        theta = 360 + math.atan(y/x)*180/math.pi #Q4   
        if (theta == 360): theta = 0
        
    if option !=0: # theta goes from -90° to 270°
        if theta >=270: theta = theta - 360
        
    return theta,phi

def spher2cart(theta,phi,R,option=1):
    """
    This function transforms Cartesian coordinates into spherical coordinates
                     
    Parameters
    ----------
     - theta
     - phi
     - R
     - option: 0 ==> angle in [°], any other value ==> angle in [rad]
    """
    if (option == 0): # transform [°] to [rad]
        theta = theta/180*math.pi
        phi = phi/180*math.pi
#        print('angle in [°]')        
    x = R*math.sin(phi)*math.cos(theta)
    y = R*math.sin(phi)*math.sin(theta)
    z = R*math.cos(phi)
#    check = math.sqrt(x**2 + y**2 + z**2)
#    print('R vs. Check :',R,check)
    return x,y,z 

def angle2dose(theta,phi):
    # Temporaire, à améliorer
#    if phi < 60: D0 = 10000 - phi*10000/60 # [Gy/h * m²/mA]
#    else : D0 = 0
#    mu0 = 0.05 # [g/cm²]
    if theta > 180: theta = 360 - theta
    if theta > 90: theta = 180 - theta
    i = getIndex(theta,10,9)[0]
    j = getIndex(phi,5,18)[0]
    x = [ 2.5,  7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5,
           57.5, 62.5, 67.5, 72.5, 77.5, 82.5, 87.5]
    D = np.zeros((9,18))
    D[0,:] = [9932,6942,5815,4635,3801,3393,2711,2316,2122,1881,1718,1375,1237,1014,802,572,306,53]
    D[1,:] = [9932,6942,5815,4635,3801,3393,2711,2316,2313,1983,1681,1477,1286,1015,717,324,80 ,31]
    D[2,:] = [9932,6942,5815,4635,3801,3393,2711,2316,2322,1932,1719,1406,1111,669 ,245,91 ,58 ,30]
    D[3,:] = [9932,6942,5815,4635,3801,3393,2711,2316,2224,1992,1567,1103,654 ,265 ,113,83 ,63 ,33]
    D[4,:] = [9932,6942,5815,4635,3801,3393,2711,2316,2093,1738,1266,796 ,348 ,157 ,105,78 ,57 ,34]
    D[5,:] = [9932,6942,5815,4635,3801,3393,2711,2316,2058,1603,1027,517 ,283 ,184 ,135,106,90 ,48]
    D[6,:] = [9932,6942,5815,4635,3801,3393,2711,2316,1906,1281,757	,413 ,229 ,176 ,130,104,87 ,49]
    D[7,:] = [9932,6942,5815,4635,3801,3393,2711,2316,1810,1067,648	,348 ,223 ,158 ,126,98 ,81 ,50]
    D[8,:] = [9932,6942,5815,4635,3801,3393,2711,2316,1696,982 ,575	,304 ,220 ,156 ,117,96 ,75 ,48]

    D0 = D[i,j] #[Gy/h * m²/mA]
    
    mu0 = 0.031 + 0.005/60*phi # [g/cm²]
    
    return D0, mu0

#def crossBox(A,B,Box[0],Box[1],option):
def crossBox(P, Box, option, decim):
    """
    This function computes the intersection points between a segment of line and a box.
    If an extremity of the segment touches a face of the box, crossBox considers the intersection occurs.
    If the segment is inside the box without crossing any of its faces, # of intersection = 0.
                     
    Parameters
    ----------
    P : 2x3 array, that contains A & B, the two points defining the line.
    Box: 2x3 array, that contains Box[0] and Box[1], the description of the box:
        - either the center and the size (option = 0)
        - or the two corners (option = 1).
    option : 0 or 1. Type of definition of the box (center & size or corners).
    decim : # of decimals. Should be < 8 to avoid issue (rounding limitation of Python).

    Returns
    -------
    nIntercept : integer. Number of intersections.
    Intersection: (6,3) Array containing the coordinates of the intersections.
    """

    Intersection = np.zeros((6, 3))
    nIntercept = 0
    epsilon = 0.0001
    # Rounding the array to avoid error when point on the line.
    # Ex: inIsIntercepting function could not find an intercept (ex: 40.000000012 < 40.0 is untrue)
    P = np.round(P, decim)
    Box = np.round(Box, decim)
    
    Corner = np.array([[0, 0, 0], [0, 0, 0]])
    if option == 0:  # transform center & size ==> corners
        Corner[0] = Box[0] - 0.5 * Box[1]
        Corner[1] = Box[0] + 0.5 * Box[1]
    else:
        Corner[0] = Box[0]
        Corner[1] = Box[1]

    unique_intersections = set()
    
    for j in range(3):  # loop over 3 pairs of box faces
        if P[1][j] != P[0][j]:  # check that the line is not parallel to the box faces
            for i in range(2):  # loop over the 2 faces of the pair
                lambda1 = (Corner[i][j] - P[0][j]) / (P[1][j] - P[0][j])
                D1 = np.round(P[0] + lambda1 * (P[1] - P[0]), decim)  # Intersection with the plane
                if isIntercepting(D1, P[0], P[1], Corner[0], Corner[1]) == 3:  # one intersection found with the box
                    D1_tuple = tuple(D1)
                    if D1_tuple not in unique_intersections:
                        unique_intersections.add(D1_tuple)
                        Intersection[nIntercept] = D1
                        nIntercept += 1
    
    return nIntercept, Intersection

def isIntercepting(X, A, B, Box1, Box2):
    """
    This function checks that the intersection points X is on the line segment and within the box.
                     
    Parameters
    ----------
    X : 1x3 array, the intersection point.
    A & B: 1x3 arrays, the two points defining the segment of line.
    Box1 & Box2: 1x3 arrays, the two corners of the box.

    Returns
    -------
    index : # of satisfied conditions. If == 3 ==> all conditions are satisfied.
    """
    index = 0
    for i in range(3):
        if Box1[i] <= X[i] <= Box2[i]:  # check that the point is within the box
            if A[i] < B[i]:
                if A[i] <= X[i] <= B[i]:  # check that the point is between A and B
                    index += 1
            else:
                if B[i] <= X[i] <= A[i]:  # check that the point is between B and A
                    index += 1
    return index

    

def dist2points(A,B):
    d = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2 + (A[2]-B[2])**2)
    return d

def absorbtionFunc(x, a, D0, Mu0):
    """
    This function describes the depth dose curve of X-rays in matter when the 
    dose is computed in Kerma water.
    The function is divided in two parts:
        - From 0 to x0 (~3.8 g/cm²): f1(x) = a*x² + b*x + c
        - From x0 to xmax: f2(x) = D0*exp(-Mu0*x)
    with the 2 constraints:
        f1(X0) = f2(x0)
        f1'(X0) = f2'(x0)
                 
    Parameters
    ----------
    - x is the array containing the depth points
    - a [cm4/g²] 
    - D0 [dose unit]
    - Mu0 [cm²/g]
    """
    
    x0 = intersectionPoint() # [g/cm²]. X0 is the intersection between the poly(2) fit and the exponential fit
    i0, x0 = find_nearest(x, x0) 

    #x0 = x[i0] # [g/cm²]
    
    expneg = np.exp(-1*Mu0*x0)
    b = -1*Mu0*D0*expneg - 2*a*x0
    c = D0*expneg - a*x0**2 - b*x0

    imax = len(x)
    y = np.zeros(imax)
    
    for i in range(i0): y[i] = a*x[i]**2 + b*x[i] + c
    for i in range(i0,imax): y[i] = D0*np.exp(-Mu0*x[i]) 

    return y

def intersectionPoint():
    """
    This function returns the intersection point for the fitting of X-rays depth dose curves.
    cfr. DepthDoseFitting.xlsx
    """
    return 7.5 # [g/cm²]

def mapping2spher(pos,dim,voxSize):
    """
    This function looks for the extremity of the mapping in spherical coordinates. It  computes (theta,phi) for 
        - the 8 corners of the mapping
        - 2 middle points
    """
    nPoints = 10 # 8 corners + 2 points for  phi_min & phi_max
    theta = np.zeros(nPoints)
    phi = np.zeros(nPoints)
    #Corner = np.zeros(3)
    xmin = pos[0] + 0.5*(-dim[0] + voxSize[0])
    ymin = pos[1] + 0.5*(-dim[1] + voxSize[1])
    zmin = pos[2] + 0.5*(-dim[2] + voxSize[2])
    
    xmax = pos[0] + 0.5*(dim[0] - voxSize[0])
    ymax = pos[1] + 0.5*(dim[1] - voxSize[1])
    zmax = pos[2] + 0.5*(dim[2] - voxSize[2])

    xmina = absMin(pos[0] + 0.5*(-dim[0] + voxSize[0]),
                  pos[0] + 0.5*(dim[0] - voxSize[0]))
    ymina = absMin(pos[1] + 0.5*(-dim[1] + voxSize[1]),
                  pos[1] + 0.5*(dim[1] - voxSize[1]))                                
    for i in range(nPoints):
        if i == 0:
            Corner = pos - 0.5*dim + 0.5*voxSize
        elif i == 1: Corner[:] = [xmax, ymin, zmin]
        elif i == 2: Corner[:] = [xmin, ymax, zmin]
        elif i == 3: Corner[:] = [xmax, ymax, zmin]
        elif i == 4: Corner[:] = [xmin, ymin, zmax]
        elif i == 5: Corner[:] = [xmax, ymin, zmax]
        elif i == 6: Corner[:] = [xmin, ymax, zmax]
        elif i == 7:
            Corner = pos + 0.5*dim - 0.5*voxSize
        elif i == 8: # look for max Phi (ex: Phi = 90)
            Corner[:] = [xmax,ymina,zmin]            
        else: # look for min Phi (ex: Phi = 0)
            Corner[:] = [xmina,ymina,zmax]

        theta[i],phi[i] = cart2spher(Corner[0],Corner[1],Corner[2],1)
        print('Corner[i], theta, phi',i,np.round(Corner),round(theta[i],1),round(phi[i],1))
#    print('theta',theta)
#    print('phi',phi)
    return min(theta),max(theta),min(phi),max(phi)
            
def absMin(xmin,xmax):  
#    if option < 0 # look for absolute minimum
    if xmin < 0 and xmax > 0: xAbsMin = 0
    else: xAbsMin = xmin
    
    return xAbsMin

def printAndWrite(s,f):
    print(s)
    f.write("\n"+s+"\n")
    
def writeInLine(arr,f): 
    """
    This function write the array arr in line (source: OpenAI)
    """
    line_length = 100
    for iarr in range(arr.shape[0]):
        line = ''
        # Convert the value to a string and append it to the line
        val_str = str(arr[iarr])
        line += val_str + ','
        # Remove the trailing comma
        line = line[:-1] + ' '
        while len(line) > line_length:
            f.write(line[:line_length] + '\n')
            line = line[line_length:]
        f.write(line)

    f.write("\n")  

def getWidth(x,y,y_left,y_right,slope = 'inc'):
    """
    This function returns the width of a Gaussian-like or parabolic like profile:
        - If y_left = y_right, the function returns the width at y_left
        - If y_left != y_right, the function compute the distance between y_left and y_right. In this
            case, one should specific correcty if the function is increasing or decreasing using 'slope' variable.
         - y must be a numpy array!!!  
    """
    # !!! x & y must  be an array, not a list!!!!
    res = len(x)
    
    if len(y) != res: print('WARNING: X and Y have different size')
    
    if y_left == y_right:
        x_left = np.interp(y_left,y[0:int(0.5*res)],x[0:int(0.5*res)])
        x_right = np.interp(-1*y_right,-1*y[int(0.5*res):res],x[int(0.5*res):res])
        width = x_right - x_left
        
    else :
        if slope == 'inc': # increasing function
           # x1_left = np.interp(y_left,y[0:int(0.5*res)],x[0:int(0.5*res)])
           # x2_left = np.interp(y_right,y[0:int(0.5*res)],x[0:int(0.5*res)])
            x1_left = np.interp(y_left,y,x)
            x2_left = np.interp(y_right,y,x)
            #width_left = x2_left - x1_left
            width = x2_left - x1_left
        else: # decreasing function
            x1_right = np.interp(-1*y_left,-1*y,x)
            x2_right = np.interp(-1*y_right,-1*y,x)
            #width_right = x1_right - x2_right
            width = x1_right - x2_right
        
#        width = np.mean([width_left, width_right])
        
    return width

def Nfactor(SW, sigma):
    """
    This function returns the value of factor N t oretrieve the inflection point (cfr. M-ID 119414, slide 16) :
        units: [cm]

    """
    N = 1e-5*SW*SW - 0.0061*SW + 2.9082
    
    return N

def penMark(x,Dose,threshold,index_MarkSize,direction = 'end'):
    """
    This function returns the position of the pen Mark on a CTA
        'end': look for the pen from the end of the function (typically for the beginning of the CTA)


    """
    dDose_dx = abs(np.gradient(Dose, x))
    #print(dDose_dx[0:100])
    
    if direction == 'beginning':
        # Find the first occurrence where a value exceeds the threshold
        exceed_index = np.argmax(dDose_dx > threshold)
    else:
        # Find the last occurrence where a value exceeds the threshold
        exceed_index = np.argmax(dDose_dx[::-1] > threshold)

        # Correct for the last occurrence index since we reversed the array
        exceed_index = len(dDose_dx) - exceed_index - 1
#        print("exceed index = ",exceed_index,index_MarkSize)
#        print(Dose[exceed_index - index_MarkSize : exceed_index + index_MarkSize])
        
    #print("threshold & index:",threshold,exceed_index,index_MarkSize)
    #print(Dose)
    mark_index = np.argmax(Dose[exceed_index - index_MarkSize : exceed_index + index_MarkSize])
    mark_index = mark_index+exceed_index - index_MarkSize
    #print(Dose[mark_index])
    
    return mark_index