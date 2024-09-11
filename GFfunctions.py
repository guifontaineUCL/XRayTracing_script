import matplotlib.pyplot as plt
import numpy as np
import DPfunctions as DP
import math
import sys
import pickle
from scipy.interpolate import griddata
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import KDTree

def parse_multiline_array(lines, start_index):
    """
    Parses a multiline array of floating-point numbers from a list of lines.

    Parameters:
        lines (list of str): List of lines from the input file.
        start_index (int): The index in lines where the array starts.

    Returns:
        array (list of float): Parsed array of floating-point numbers.
        next_index (int): The index of the next line after the parsed array.
    """
    array = []
    i = start_index
    while i < len(lines):
        line = lines[i].strip()
        if line.endswith(']'):
            array.extend(map(float, line.strip('[]').split()))
            break
        else:
            array.extend(map(float, line.strip('[],').split()))
        i += 1
    return array, i + 1

def parse_multiline_positions(lines, start_index):
    """
    Parses a multiline array of positions (tuples of floating-point numbers) from a list of lines.

    Parameters:
        lines (list of str): List of lines from the input file.
        start_index (int): The index in lines where the positions array starts.

    Returns:
        positions (list of list of float): Parsed positions as a list of lists.
        next_index (int): The index of the next line after the parsed positions.
    """
    positions = []
    i = start_index
    while i < len(lines):
        line = lines[i].strip()
        if line.endswith(']]'):
            positions.append(list(map(float, line.strip('[]').split())))
            break
        else:
            positions.append(list(map(float, line.strip('[],').split())))
        i += 1
    return positions, i + 1

def parse_input_file(file_path):
    """
    Parses an input file to extract the 'Alpha', 'Positions', and 'Lengths' arrays.

    The function handles both single-line and multiline arrays. It returns the parsed arrays
    as NumPy arrays.

    Parameters:
        file_path (str): The path to the input file to be parsed.

    Returns:
        alpha (np.ndarray): Parsed 'Alpha' array.
        positions (np.ndarray): Parsed 'Positions' array.
        lengths (np.ndarray): Parsed 'Lengths' array.
    """
    alpha = []
    positions = []
    lengths = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Find and parse the alpha values
        next_index = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('Alpha = ['):
                alpha_line = line.strip().lstrip('Alpha = ')
                # Check if the array is on a single line
                if alpha_line.endswith(']'):
                    alpha.extend(map(float, alpha_line.strip('[]').split()))
                    next_index = i + 1
                else:
                    alpha.extend(map(float, alpha_line.strip('[],').split()))
                    alpha_additional, next_index = parse_multiline_array(lines, i + 1)
                    alpha.extend(alpha_additional)
                break

        # Find and parse the positions
        for i in range(next_index, len(lines)):
            if lines[i].strip().startswith('Positions of the sources: ['):
                positions_line = lines[i].strip().lstrip('Positions of the sources: ')
                # Check if the array is on a single line
                if positions_line.endswith(']]'):
                    positions.append(list(map(float, positions_line.strip('[]').split())))
                    next_index = i + 1
                else:
                    positions.append(list(map(float, positions_line.strip('[],').split())))
                    positions_additional, next_index = parse_multiline_positions(lines, i + 1)
                    positions.extend(positions_additional)
                break

        # Find and parse the lengths
        for i in range(next_index, len(lines)):
            if lines[i].strip().startswith('Lengths of the sources: ['):
                lengths_line = lines[i].strip().lstrip('Lengths of the sources: ')
                # Check if the array is on a single line
                if lengths_line.endswith(']'):
                    lengths.extend(map(float, lengths_line.strip('[]').split()))
                    next_index = i + 1
                else:
                    lengths.extend(map(float, lengths_line.strip('[],').split()))
                    lengths_additional, next_index = parse_multiline_array(lines, i + 1)
                    lengths.extend(lengths_additional)
                break

    return np.array(alpha), np.array(positions), np.array(lengths)


# Define the exponential function
def exponential(x, d, e):
    return d * np.exp(-e * x)

# Define the quadratic function
def quadratic(x, a, d, e, x0):
    b = -2*a*x0 - d*e*np.exp(-e * x0)
    c = -a*x0**2 - b*x0 + d*np.exp(-e * x0)
    return a * x ** 2 + b * x + c

def combined_func(x, a, d, e, x0):
    if x <= x0:
        return quadratic(x, a, d, e, x0)
    else:
        
       return exponential(x, d, e)
   
def function(theta,phi,depth,thetas,phis,depths,doses):
    points = np.array([thetas, phis, depths]).T  # shape should be (n_points, 3)
    result = griddata(points, doses, (theta,phi,depth), method='linear')
    if np.isnan(result):
        return 0.0
    else:
        return result

def compute_dose(Energy, Method, PointInProduct, boxes, n_boxes, densities, source_cell_coord, source_cell_weights, units_num, current, charge):
    """
    Computes the dose at a given point in the product based on X-ray source parameters and product geometry.

    Parameters:
    Energy (float): The energy of the X-ray source, typically in keV or MeV.
    Method (str): The method used to compute the dose (e.g., "Angle2Dose", "RBF", etc.).
    PointInProduct (numpy array): The coordinates of the point in the product where the dose is to be computed.
    boxes (list of numpy arrays): List of bounding boxes for different regions in the product.
    n_boxes (int): Number of boxes (regions) in the product.
    densities (list of floats): List of densities for each region in the product.
    source_cell_coord (numpy array): Coordinates of the source cells.
    source_cell_weights (numpy array): Weight coefficients for the source cells.
    units_num (int): Unit conversion identifier (0 for Gy/h, 1 for kGy/h, 2 for Gy, 3 for kGy).
    current (float): The current of the X-ray source, typically in amperes (A).
    charge (float): The total charge delivered by the source, typically in coulombs (C).

    Returns:
    float: The computed dose at the given point in the product in the specified units.
    """

    Dose = 0.0
    PointInProduct_z = PointInProduct[2]
    PointInProduct_x = PointInProduct[0]
    PointInProduct_y = PointInProduct[1]

    # Precompute repetitive values
    current_coeff = 0.0
    if units_num == 0: 
        current_coeff = current # [Gy/h]
    elif units_num == 1: 
        current_coeff = 0.001 * current # [kGy/h]
    elif units_num == 2: 
        current_coeff = charge # [Gy]
    else: 
        current_coeff = 0.001 * charge # [kGy]

    #with open('linear_interpolator_2.pkl', 'rb') as f:
        #interpolator = pickle.load(f)
    
    if Method == "RBF":
        with open(f'./Methods_data/RBF_Method_Data_{Energy}MeV.pkl', 'rb') as f:
            rbf_x0, rbf_a, rbf_d, rbf_e = pickle.load(f)
    if Method == "Linear":
        # Load the precomputed data
        with open(f'./Methods_data/Linear_Method_Data_{Energy}MeV.pkl', 'rb') as f:
            precomputed_data = pickle.load(f)
        # Extract the keys (phi, theta, depth) and the corresponding values
        keys = np.array(list(precomputed_data.keys()))
        values = np.array(list(precomputed_data.values()))
        # Build a KDTree for efficient nearest-neighbor search
        kdtree = KDTree(keys)
    
    # Loop through each source cell
    for i in range(len(source_cell_coord)):
        x_source, y_source = source_cell_coord[i]
        
        xd = x_source - PointInProduct_x  # [cm]
        yd = y_source - PointInProduct_y  # [cm]
        
        # Convert cartesian to spherical coordinates
        theta, phi = DP.cart2spher(xd, yd, PointInProduct_z)
        # for easiness
        if theta > 180: theta = 360 - theta
        if theta > 90: theta = 180 - theta
            
        # Calculate the initial dose and attenuation coefficient
        if Method == "RBF":
            x0, a, D0, mu0 = rbf_x0(theta, phi), rbf_a(theta, phi), rbf_d(theta, phi), rbf_e(theta, phi)
        if Method == "Angle2Dose":
            if Energy == 7:
                D0, mu0 = DP.angle2dose_7MeV(theta, phi)  # [Gy/h * m²/mA] , [cm²/g]
            else:
                print(f"Error: '{Energy}MeV' not implemented for Angle2Dose method")
        
        # Compute the distance from the source to the point
        r = math.sqrt(xd**2 + yd**2 + PointInProduct_z**2)  # [cm]
        
        # Prepare the line points for intersection calculations
        P = np.array([[x_source, y_source, 0], PointInProduct])  # Line defined by two points
        
        # Calculate intersections and distances for each box
        total_distance_density = 0.0
        for t in range(n_boxes):
            nIntercept, Intersection = DP.crossBox(P, boxes[t], 0, 2)
            
            if nIntercept == 0 and t == 0:
                raise ValueError("ERROR: no intersection! Point in product: {}, Box: {}".format(PointInProduct, boxes[0]))
            
            if nIntercept == 2:
                d = DP.dist2points(Intersection[0], Intersection[1])
                total_distance_density += d * densities[t]

            elif nIntercept == 1 and t == 0:
                d = DP.dist2points(PointInProduct, Intersection[0])
                total_distance_density += d * densities[0]

        # Compute dose contribution from this source cell
        if Method == "Angle2Dose":
            exp_term = math.exp(-mu0 * total_distance_density)
            Dose += source_cell_weights[i] * D0 * exp_term / (0.01 * r)**2  # [Gy/mAh]
        elif Method == "RBF":
            term = combined_func(total_distance_density, a, D0, mu0, x0)
            Dose += source_cell_weights[i] * term / (0.01 * r)**2  # [Gy/mAh]
        elif Method == "Linear":
            query_point = np.array([theta, phi, total_distance_density])
            dist, idx = kdtree.query(query_point)
            term = values[idx]
            if np.isnan(term):
                term = 0
            Dose += source_cell_weights[i] * term / (0.01 * r)**2  # [Gy/mAh]

    # Apply the current or charge coefficient to convert the dose
    Dose *= current_coeff
    
    return Dose

def plot_heatmaps(doses, exp_num, units, box_center, box_dim, bool_double):
    """
    Plots heatmaps of dose distributions for three XY-planes and saves the plots as an image file.

    Parameters:
    doses (numpy array): 3D array of doses to be plotted for each of the three XY-planes.
    exp_num (int): Experiment number used for labeling the output file.
    units (str): Units of the dose to be displayed on the color bar.
    box_center (numpy array): The center of the box (length 3 array).
    box_dim (numpy array): The dimensions of the box (length 3 array).
    bool_double (bool): Flag indicating whether the dose is from double-side irradiation.

    Returns:
    None
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    planes = ['front', 'middle', 'back']

    for i in range(3):
        ax = axes[i]
        cax = ax.imshow(np.flip(doses[i],axis=0), cmap='hot', interpolation='nearest', extent=[
                        box_center[1] - box_dim[1] / 2, box_center[1] + box_dim[1] / 2,
                        box_center[0] - box_dim[0] / 2, box_center[0] + box_dim[0] / 2])
        ax.set_title(f'{planes[i]} X-Y plane dose distribution')
        ax.set_xlabel('Y coordinate')
        ax.set_ylabel('X coordinate')
        fig.colorbar(cax, ax=ax, label=f'Dose ({units})')
       
    plt.tight_layout()
    if bool_double:
        plt.savefig(f"./Saved_figures/Experiment_{exp_num+1}_front-middle-back_XY_dose_planes_double_side.png")
    else:
        plt.savefig(f"./Saved_figures/Experiment_{exp_num+1}_front-middle-back_XY_dose_planes_single_side.png")
        
def find_min_max_dose_3planes(func,central_prod_res,*args):
    """
    Computes the minimum and maximum dose in three XY-planes of a discretized box.

    Parameters:
    func (function): A function that takes a point (numpy array) and additional arguments (*args) to compute the dose at that point.
    central_prod_res (tuple): A tuple containing the resolution (dx, dy, dz) for the central product region.
    *args: Additional arguments required by func. 
        - args[0][0][0]: The center of the box (numpy array of length 3).
        - args[0][0][1]: The dimensions of the box (numpy array of length 3).
        - other arguments relevant for func

    Returns:
    tuple: 
        - doses_single (numpy array): 3D array of doses for each point in the three XY-planes.
        - doses_double (numpy array): 3D array of doses considering double-sided irradiation.
        - doses_points (numpy array): 3D array of points corresponding to the doses in the three XY-planes.
    """
    box_center = args[0][0][0]
    box_dim = args[0][0][1]
    
    nx = int(box_dim[0]/central_prod_res[0])
    ny = int(box_dim[1]/central_prod_res[1])
    
    doses = np.empty((3,nx,ny))
    doses_points = np.empty((3,nx,ny),dtype=object)
    planes = ["front","middle","back"]
    
    for ind_z in range(3):
        print(f"Searching {planes[ind_z]} XY-plane ...")
        z = box_center[2]+(box_dim[2]-central_prod_res[2])/2*(ind_z-1)
        for ind_x in range(nx):
            x = box_center[0]-box_dim[0]/2+(ind_x+0.5)*central_prod_res[0]
            for ind_y in range(ny):
                y = box_center[1]-box_dim[1]/2+(ind_y+0.5)*central_prod_res[1]
                point = np.array([x,y,z])
                doses[ind_z,ind_x,ind_y] = func(point,*args)
                doses_points[ind_z,ind_x,ind_y] = point
    
    doses_single = doses.copy()
    doses_double = doses[:,:,:] + doses[::-1,:,:]
    
    return doses_single, doses_double, doses_points

def find_min_max_dose_9lines(func,central_prod_res,*args):
    """
    Computes the minimum and maximum dose along 9 vertical lines : 4 (Y,Z) corners, 4 (Y,Z) mid vertical faces and (Y,Z) center.

    Parameters:
    func (function): A function that takes a point (numpy array) and additional arguments (*args) to compute the dose at that point.
    central_prod_res (tuple): A tuple containing the resolution (dx, dy, dz) for the central product region.
    *args: Additional arguments required by func. 
        - args[0][0][0]: The center of the box (numpy array of length 3).
        - args[0][0][1]: The dimensions of the box (numpy array of length 3).
        - other arguments relevant for func

    Returns:
    tuple: 
        - doses_single (numpy array): 2D array of doses for each point in the three XY-planes.
        - doses_double (numpy array): 2D array of doses considering double-sided irradiation.
        - doses_points (numpy array): 2D array of points corresponding to the doses in the three XY-planes.
    """
    box_center = args[0][0][0]
    box_dim = args[0][0][1]
    
    nx = int(box_dim[0]/central_prod_res[0])
    
    doses = np.empty((9,nx))
    doses_points = np.empty((9,nx),dtype=object)
    YZ_pos = np.array([[box_center[1], box_center[2]],
                       [box_center[1], box_center[2]-box_dim[2]/2+central_prod_res[2]/2],
                       [box_center[1], box_center[2]+box_dim[2]/2-central_prod_res[2]/2],
                       [box_center[1]-box_dim[1]/2+central_prod_res[1]/2, box_center[2]],
                       [box_center[1]+box_dim[1]/2-central_prod_res[1]/2, box_center[2]],
                       [box_center[1]-box_dim[1]/2+central_prod_res[1]/2, box_center[2]-box_dim[2]/2+central_prod_res[2]/2],
                       [box_center[1]-box_dim[1]/2+central_prod_res[1]/2, box_center[2]+box_dim[2]/2-central_prod_res[2]/2],
                       [box_center[1]+box_dim[1]/2-central_prod_res[1]/2, box_center[2]-box_dim[2]/2+central_prod_res[2]/2],
                       [box_center[1]+box_dim[1]/2-central_prod_res[1]/2, box_center[2]+box_dim[2]/2-central_prod_res[2]/2]])
    
    for i in range(len(YZ_pos)):
        print(f"Searching line number {i+1} out of 9 ...")
        y,z = YZ_pos[i]
        for ind_x in range(nx):
            x = box_center[0]-(box_dim[0]-central_prod_res[0])/2 + central_prod_res[0]*ind_x
            point = np.array([x,y,z])
            doses[i,ind_x] = func(point,*args)
            doses_points[i,ind_x] = point
    
    doses_single = doses.copy()
    doses_double = doses[:,:] + np.array([doses[0,:],doses[2,:],doses[1,:],doses[3,:],doses[4,:],doses[6,:],doses[5,:],doses[8,:],doses[7,:]])

    return doses_single, doses_double, doses_points

def find_min_max_dose_6lines(func,central_prod_res,*args):
    """
    Computes the minimum and maximum dose along 6 vertical lines : 2 (Y,Z) corners, 3 (Y,Z) mid vertical faces and (Y,Z) center.

    Parameters:
    func (function): A function that takes a point (numpy array) and additional arguments (*args) to compute the dose at that point.
    central_prod_res (tuple): A tuple containing the resolution (dx, dy, dz) for the central product region.
    *args: Additional arguments required by func. 
        - args[0][0][0]: The center of the box (numpy array of length 3).
        - args[0][0][1]: The dimensions of the box (numpy array of length 3).
        - other arguments relevant for func

    Returns:
    tuple: 
        - doses_single (numpy array): 2D array of doses for each point in the three XY-planes.
        - doses_double (numpy array): 2D array of doses considering double-sided irradiation.
        - doses_points (numpy array): 2D array of points corresponding to the doses in the three XY-planes.
    """
    box_center = args[0][0][0]
    box_dim = args[0][0][1]
    
    nx = int(box_dim[0]/central_prod_res[0])
    
    doses = np.empty((6,nx))
    doses_points = np.empty((6,nx),dtype=object)
    YZ_pos = np.array([[box_center[1], box_center[2]],
                       [box_center[1], box_center[2]-box_dim[2]/2+central_prod_res[2]/2],
                       [box_center[1], box_center[2]+box_dim[2]/2-central_prod_res[2]/2],
                       [box_center[1]-box_dim[1]/2+central_prod_res[1]/2, box_center[2]],
                       [box_center[1]-box_dim[1]/2+central_prod_res[1]/2, box_center[2]-box_dim[2]/2+central_prod_res[2]/2],
                       [box_center[1]-box_dim[1]/2+central_prod_res[1]/2, box_center[2]+box_dim[2]/2-central_prod_res[2]/2]])
    
    for i in range(len(YZ_pos)):
        print(f"Searching line number {i+1} out of 6 ...")
        y,z = YZ_pos[i]
        for ind_x in range(nx):
            x = box_center[0]-(box_dim[0]-central_prod_res[0])/2 + central_prod_res[0]*ind_x
            point = np.array([x,y,z])
            doses[i,ind_x] = func(point,*args)
            doses_points[i,ind_x] = point
    
    doses_single = doses.copy()
    doses_double = doses[:,:] + np.array([doses[0,:],doses[2,:],doses[1,:],doses[3,:],doses[5,:],doses[4,:]])

    return doses_single, doses_double, doses_points

def save_log(file_path, line_to_write, should_create_file, should_print):
    """
    Creates a file and writes a line to it if the boolean argument is True.
    
    Parameters:
    should_create_file (bool): Flag to determine whether to create the file and write to it.
    file_path (str): The path of the file to create.
    line_to_write (str): The line to write to the file.
    """
    if should_create_file:
        try:
            with open(file_path, 'w') as file:
                file.write(line_to_write + '\n')
                if should_print:
                    print(line_to_write)
        except Exception as e:
            print(f"An error occurred while creating or writing to the file: {e}")
    else:
        try:
            with open(file_path, 'a') as file:
                file.write(line_to_write + '\n')
                if should_print:
                    print(line_to_write)
        except Exception as e:
            print(f"An error occurred while writing to the file: {e}")

def compute_source_cell_info(bool_variable, x_variable_bool, source_xRes, source_yRes, width_scan, length_nom, positions, lengths, alphas):         
    """
    Computes source cell coordinates and weight coefficients based on various input conditions.

    Parameters:
    bool_variable (bool): A condition that determines how to calculate the source cell information.
    x_variable_bool (bool): A condition that specifies the calculation mode, likely related to the X-axis.
    source_xRes (float): The resolution of the source in the X direction.
    source_yRes (float): The resolution of the source in the Y direction.
    width_scan (float): The width of the scan area.
    length_nom (float): The nominal length of the source.
    positions (list of tuples): A list of tuples specifying the positions of sources or cells.
    lengths (list of floats): A list of lengths associated with each position.
    alphas (list of floats): A list of alpha values related to each source or cell.

    Returns:
    tuple: A tuple containing two numpy arrays:
        - source_cell_coord (numpy array): The computed coordinates of the source cells.
        - source_cell_weights (numpy array): The computed weight coefficients for the source cells.
    """
    nSource_X_cells = int(width_scan/source_xRes)
    nSource_Y_cells = int(length_nom/source_yRes)
    if not bool_variable:
        source_cell_coord = np.array([
            ((l + 0.5) * source_xRes - 0.5 * width_scan, 
            (m + 0.5) * source_yRes - 0.5 * length_nom) 
            for m in range(nSource_Y_cells) 
            for l in range(nSource_X_cells)
        ])
        source_cell_weights = np.array([source_xRes * source_yRes / (width_scan * length_nom) 
            for _ in range(nSource_Y_cells) 
            for _ in range(nSource_X_cells)
        ])
        
    elif x_variable_bool:
        source_cell_coord = []
        source_cell_weights = []
        for i, pos in enumerate(positions):
            pos_x = pos[0]
            l = lengths[i]
            n_intervals = int(np.ceil(l / source_xRes))
            dx = l / n_intervals
            alpha = alphas[i]
            for j in range(n_intervals):
                for k in range(nSource_Y_cells):
                    source_cell_coord.append(
                        (pos_x - l / 2 + dx / 2 + dx * j, 
                        (k + 0.5) * source_yRes - 0.5 * length_nom)
                    )
                    source_cell_weights.append(dx * source_yRes * alpha / (length_nom * sum(lengths)))
        source_cell_coord = np.array(source_cell_coord)
        source_cell_weights = np.array(source_cell_weights)
    else:
        source_cell_coord = []
        source_cell_weights = []
        for i, pos in enumerate(positions):
            pos_y = pos[1]
            l = lengths[i]
            n_intervals = int(np.ceil(l / source_yRes))
            dy = l / n_intervals
            alpha = alphas[i]
            for j in range(n_intervals):
                for k in range(nSource_X_cells):
                    source_cell_coord.append(
                        ((k + 0.5) * source_xRes - 0.5 * width_scan, 
                        pos_y - l / 2 + dy / 2 + dy * j)
                    )
                    source_cell_weights.append(dy * source_xRes * alpha / (width_scan * sum(lengths)))
        
        source_cell_coord = np.array(source_cell_coord)
        source_cell_weights = np.array(source_cell_weights)
       
    return source_cell_coord, source_cell_weights

