# -*- coding: utf-8 -*-
"""
XRayTracing_script.py

Created on Fri Jul 19 

@author: gf
@author: Guilluame Fontaine
@script version: 1.0

The `XRayTracing_script.py` is a Python script designed to simulate the dose distribution of X-ray radiation on a central product from a radiation source. The simulation considers both source-related parameters (such as energy-angle spectrum and geometry) and system parameters (including geometry and density) for the conveyor, as well as the left, right, and central products, wooden pallets, and mother pallets.

The principal objective of this script is to provide a faster alternative to Monte Carlo simulations, such as those performed by the RayXpert tool, which are often too slow for practical use. This script aims to achieve an optimal balance between speed and accuracy.

The simulation has three primary objectives:
1. **Compute 1D Dose Mappings:** Calculate and plot dose profiles along the X, Y, and Z directions around specific 1D mapping points.
2. **Compute 0D Dose Mappings:** Determine the dose at discrete X, Y, Z points (0-D mappings).
3. **Extract Dose Metrics:** Evaluate the minimum dose, maximum dose, and dose uniformity ratio (DUR) within the central product for both single and double irradiation scenarios.

The script is designed to handle multiple experimental setups, which can be specified in the `XRayTracing_Input.xlsx` file.
"""

import matplotlib.pyplot as plt
import numpy as np
import DPfunctions as DP
import pandas as pd
import math
import sys
import time
import re

### Choose the pathname and files
pathname = "" #r'C:\Users\dp\OneDrive - IBA Group\My Documents - Operationnel\1. My Projects\Y21 RayXpert\Y22_03 - XR Source Definition'
inputFile = 'XRayTracing_Input.xlsx' #pathname + '\\' + 'XRayTracing.inp'

#######################################################################
################### Function definitions ##############################
#######################################################################

def compute_dose(PointInProduct,boxes,n_boxes,densities,nSource_X_cells,nSource_Y_cells,source_xRes,source_yRes,width_scan,length_nom,units_num,bool_current_x,bool_current_y,currents):
    """
    Computes the dose at a given point in the product based on X-ray source parameters and product geometry.

    Parameters:
    PointInProduct (numpy array): The coordinates of the point in the product where the dose is to be computed.
    boxes (list of numpy arrays): List of bounding boxes for different regions in the product.
    n_boxes (int): Number of boxes (regions) in the product.
    densities (list of floats): List of densities for each region in the product.
    nSource_X_cells (int): Number of source cells in the X direction.
    nSource_Y_cells (int): Number of source cells in the Y direction.
    source_xRes (float): Resolution of the source in the X direction.
    source_yRes (float): Resolution of the source in the Y direction.
    width_scan (float): Width of the scan area.
    length_nom (float): Nominal length of the source.
    units_num (int): Unit conversion identifier (0 for Gy/h, 1 for kGy/h, 2 for Gy, 3 for kGy).

    Returns:
    float: The computed dose at the given point in the product.
    """
    Dose = 0
    for l in range(nSource_X_cells): # loop over source in X (vertical)
        x_source = (l + 0.5)*source_xRes - 0.5*width_scan # [cm]
        xd = x_source - PointInProduct[0] # [cm]
        
        for m in range(nSource_Y_cells): # loop over source in Y (horizontal)
            y_source = (m + 0.5)*source_yRes - 0.5*length_nom # [cm]
            yd = y_source - PointInProduct[1] # [cm]
            theta,phi = DP.cart2spher(xd,yd,PointInProduct[2])
            D0,mu0 = DP.angle2dose(theta,phi) # [Gy/h * m²/mA] , [cm²/g]
    
            r = math.sqrt(xd**2 + yd**2 + PointInProduct[2]**2) # [cm]
            P = np.array([[x_source, y_source, 0],PointInProduct]) # the 2 points defining the line
            # check all box intersections
            nIntercept = np.empty(n_boxes,dtype=object)
            Intersection = np.empty(n_boxes,dtype=object)
            for t in range(n_boxes):
                nIntercept[t], Intersection[t] = DP.crossBox(P,boxes[t],0,2)
            
            ### Compute all distance in matter and corresponding density contributions
            distance_density = []
            ## central product
            if nIntercept[0] == 0: 
                    print("ERROR: no intersection!!!")
                    print("point on source:", [xd, yd, 0])
                    print("point in product:",PointInProduct)
                    print("The box:", Box_central_prod)
                    sys.exit()
            #elif nIntercept > 1: print("WARNING: more than one intersection!!!")
            d = DP.dist2points(PointInProduct,Intersection[0][0])
            distance_density.append([d,densities[0]])
            ## All other objects
            for t in range(1,n_boxes):
                if nIntercept[t] == 2:
                    d = DP.dist2points(Intersection[t][0],Intersection[t][1])
                    distance_density.append([d,densities[t]])
            
            # compute dose
            distance_density = np.array(distance_density)
            
            # compute current coefficient
            if bool_current_x:
                curr_weight = currents[l]/np.mean(currents)
            else:
                curr_weight = currents[m]/np.mean(currents)

            Dose += curr_weight*D0*math.exp(-mu0*sum(distance_density[q][0]*distance_density[q][1] for q in range(len(distance_density))))/(0.01*r)**2 # [Gy/mAh] 1-D array (dim = kmax) that contains the Y values of the mapping
    
    if units_num == 0: Dose *= currentPerCell # [Gy/h]
    elif units_num == 1: Dose *= 0.001*currentPerCell # [kGy/h]
    elif units_num == 2: Dose *= chargePerCell # [Gy]
    else : Dose *= 0.001*chargePerCell # [kGy]
    
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
        
def find_min_max_dose(func,central_prod_res,*args):
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


###################################################
####   Part 0: Read XRayTracing.inp        ########
###################################################

data = pd.read_excel(inputFile, header = 1)

### Loop over number of experiments
for exp_num in range(data.shape[0]):
    
    ## Only consider experiments marked by a "V" running symbol
    if data.loc[exp_num,"Run (V or X)"] == "V":
        
        # 0.0 - Set timer
        start = time.time()
        
        # 0.1 - Read source info
        length_nom = float(data.loc[exp_num, "Source Y-dimension [cm]"])

        width_scan = float(data.loc[exp_num, "Source X-dimension [cm]"])

        bool_current_x = data.loc[exp_num, "Variable Current in Y direction [mA]"] == "/"
        bool_current_y = data.loc[exp_num, "Variable Current in X direction [mA]"] == "/"
        if bool_current_x:
            current_column = data.loc[exp_num, "Variable Current in X direction [mA]"]
        else:
            current_column = data.loc[exp_num, "Variable Current in Y direction [mA]"]
        currents = np.array([float(curr) for curr in current_column.strip('[]').split(',')])

        source_xRes,source_yRes = [float(data.loc[exp_num, "Source resolution (X,Y,) [cm²]"].split("(")[1].split(",")[0]),
                                   float(data.loc[exp_num, "Source resolution (X,Y,) [cm²]"].split(",")[1].split(",")[0])]

        # 0.2 - Read Conveyor info
        conv_density = float(data.loc[exp_num, "Conveyor density [g/cm³]"])
        conv_dim = [float(data.loc[exp_num, "Conveyor dimensions [cm³]"].split("(")[1].split(",")[0]),
                    float(data.loc[exp_num, "Conveyor dimensions [cm³]"].split(",")[1].split(",")[0]),
                    float(data.loc[exp_num, "Conveyor dimensions [cm³]"].split(",")[2].split(")")[0])]
        conv_XZ_offset = [float(data.loc[exp_num, "Conveyor (X,Z,) offset to source [cm]²"].split("(")[1].split(",")[0]),
                          float(data.loc[exp_num, "Conveyor (X,Z,) offset to source [cm]²"].split(",")[1].split(",")[0])]
        conv_speed = float(data.loc[exp_num, "Conveyor speed [m/min]"])
        operation_hours = float(data.loc[exp_num, "Operation [hours/year]"])

        # 0.3 - Read central product info
        central_prod_density = float(data.loc[exp_num, "Central product density [g/cm³]"])

        central_prod_dim = [float(data.loc[exp_num, "Central product dimensions [cm³]"].split("(")[1].split(",")[0]),
                       float(data.loc[exp_num, "Central product dimensions [cm³]"].split(",")[1].split(",")[0]),
                       float(data.loc[exp_num, "Central product dimensions [cm³]"].split(",")[2].split(")")[0])]
        
        central_prod_res = [float(data.loc[exp_num, "Central product resolution (X,Y,Z) [cm³]"].split("(")[1].split(",")[0]),
                       float(data.loc[exp_num, "Central product resolution (X,Y,Z) [cm³]"].split(",")[1].split(",")[0]),
                       float(data.loc[exp_num, "Central product resolution (X,Y,Z) [cm³]"].split(",")[2].split(")")[0])]

        central_prod_yz_pos = [float(data.loc[exp_num, "Central product (Y,Z,) position [cm]"].split("(")[1].split(",")[0]),
                       float(data.loc[exp_num, "Central product (Y,Z,) position [cm]"].split(",")[1].split(",)")[0])]

        # 0.4 - Read central wooden pallet info
        central_wooden_pallet_density = float(data.loc[exp_num, "Central wooden pallet density [g/cm³]"])
        
        central_wooden_pallet_dim = [float(data.loc[exp_num, "Central wooden pallet dimensions [cm³]"].split("(")[1].split(",")[0]),
                       float(data.loc[exp_num, "Central wooden pallet dimensions [cm³]"].split(",")[1].split(",")[0]),
                       float(data.loc[exp_num, "Central wooden pallet dimensions [cm³]"].split(",")[2].split(")")[0])]
        
        # 0.5 - Read left product info
        left_prod_density = float(data.loc[exp_num, "Left product density [g/cm³]"])

        left_prod_dim = [float(data.loc[exp_num, "Left product dimensions [cm³]"].split("(")[1].split(",")[0]),
                       float(data.loc[exp_num, "Left product dimensions [cm³]"].split(",")[1].split(",")[0]),
                       float(data.loc[exp_num, "Left product dimensions [cm³]"].split(",")[2].split(")")[0])]
        
        left_prod_yGap = float(data.loc[exp_num, "Left product Y-gap [cm]"])
        
        # 0.6 - Read left wooden pallet info
        left_wooden_pallet_density = float(data.loc[exp_num, "Left wooden pallet density [g/cm³]"])
        
        left_wooden_pallet_dim = [float(data.loc[exp_num, "Left wooden pallet dimensions [cm³]"].split("(")[1].split(",")[0]),
                       float(data.loc[exp_num, "Left wooden pallet dimensions [cm³]"].split(",")[1].split(",")[0]),
                       float(data.loc[exp_num, "Left wooden pallet dimensions [cm³]"].split(",")[2].split(")")[0])]
        
        # 0.7 - Read right product info
        right_prod_density = float(data.loc[exp_num, "Right product density [g/cm³]"])

        right_prod_dim = [float(data.loc[exp_num, "Right product dimensions [cm³]"].split("(")[1].split(",")[0]),
                       float(data.loc[exp_num, "Right product dimensions [cm³]"].split(",")[1].split(",")[0]),
                       float(data.loc[exp_num, "Right product dimensions [cm³]"].split(",")[2].split(")")[0])]
        
        right_prod_yGap = float(data.loc[exp_num, "Right product Y-gap [cm]"])
        
        # 0.8 - Read right wooden pallet info
        right_wooden_pallet_density = float(data.loc[exp_num, "Right wooden pallet density [g/cm³]"])
        
        right_wooden_pallet_dim = [float(data.loc[exp_num, "Right wooden pallet dimensions [cm³]"].split("(")[1].split(",")[0]),
                       float(data.loc[exp_num, "Right wooden pallet dimensions [cm³]"].split(",")[1].split(",")[0]),
                       float(data.loc[exp_num, "Right wooden pallet dimensions [cm³]"].split(",")[2].split(")")[0])]
        
        # 0.9 - Read mother pallet info
        mother_pallet_density = float(data.loc[exp_num, "Mother pallets density [g/cm³]"])
        
        mother_pallet_dim = [float(data.loc[exp_num, "Mother pallets dimensions [cm³]"].split("(")[1].split(",")[0]),
                       float(data.loc[exp_num, "Mother pallets dimensions [cm³]"].split(",")[1].split(",")[0]),
                       float(data.loc[exp_num, "Mother pallets dimensions [cm³]"].split(",")[2].split(")")[0])]
        
        # 0.10 - Read dose info
        units_num = int(data.loc[exp_num, "Units (Gy/h) (kGy/h) (Gy) (kGy)"])
        minimum_dose = float(data.loc[exp_num, "Minimum dose"])
        
        # 0.11 - Read 0D-mappings
        XYZ_0D = []
        list = data.loc[exp_num, "List of n 0-D dose mapping points (X1,Y1,Z1), …, (Xn,Yn,Zn)"]
        if not(list == "" or (isinstance(list, float) and math.isnan(list))):
            pattern = r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)'
            # Find all tuples in the string
            matches = re.findall(pattern, list)
            # Check if each match contains positive integers
            for match in matches:
                x, y, z = map(float, match)
                XYZ_0D.append((x,y,z))
        nXY_0D = len(XYZ_0D)
        
        # 0.12 - Read 1D-mappings
        XYZ_1D = []
        list = data.loc[exp_num, "List of p 1-D mapping points (X1,Y1,Z1), …, (Xp,Yp,Zp)"]
        if not(list == "" or (isinstance(list, float) and math.isnan(list))):
            pattern = r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)'
            # Find all tuples in the string
            matches = re.findall(pattern, list)
            # Check if each match contains positive integers
            for match in matches:
                x, y, z = map(float, match)
                XYZ_1D.append((x,y,z))
        nXY_1D = len(XYZ_1D)
        
        X_mapping_bool = data.loc[exp_num, "X-mapping (V or X)"] == "V"
        Y_mapping_bool = data.loc[exp_num, "Y-mapping (V or X)"] == "V"
        Z_mapping_bool = data.loc[exp_num, "Z-mapping (V or X)"] == "V"
            
        limits = np.array([[0, 	0,	0,	0],[0, 	0,	0,	0],[0, 	0,	0,	0]])

        ########################################
        ###   Part 1: Initialize figures    ####
        ########################################
        
        if X_mapping_bool:
            fig_x, axs = plt.subplots()
        if Y_mapping_bool:
            fig_y, ays = plt.subplots() #create a new figure
        if Z_mapping_bool:
            fig_z, azs = plt.subplots()

        title = 'XRayTracing'
        #for i in range(nrxmFile): title = title + map_filename[i] + ' vs. '
        title = title[:-5]

        if units_num == 0: 
            yaxis_title = 'Dose Rate [Gy/h]' # [Gy/h]
            units = '[Gy/h]'
            minimum_dose *= 1000 # [Gy] 
        elif units_num == 1: 
            yaxis_title = 'Dose Rate [kGy/h]' # [kGy/h]
            units = '[kGy/h]'
        elif units_num == 2: 
            yaxis_title = 'Dose [Gy]' # [Gy]
            units = '[Gy]'
            minimum_dose *= 1000 # [Gy]
        else : 
            yaxis_title = 'Dose [kGy]' # [kGy]
            units = '[kGy]'

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', 
                '#7f7f7f', '#bcbd22', '#17becf','red', 'blue', 'green']

        ######################################################   
        ####  Part 2: Compute dose in product & plot  ########
        ######################################################
        
        # Compute relevant set up parameters
        nSource_X_cells = int(width_scan/source_xRes)
        nSource_Y_cells = int(length_nom/source_yRes)
        nSource_cells = nSource_X_cells * nSource_Y_cells

        mean_current = np.mean(currents)
        time_travel = 0.01*length_nom/conv_speed # [min] = Time for the pallet to go along the entire source (>< process time!)
        charge = mean_current*time_travel/60 # [mA h]
        chargePerCell = charge / nSource_cells  # [mA h]
        currentPerCell = mean_current / nSource_cells  # [mA]
        ratio_iv = mean_current/conv_speed # [mA min/m]
        pv = 6*ratio_iv/width_scan # process value [As/m²]
        nPalletPerMin = 100*conv_speed/(central_prod_dim[1] + left_prod_yGap) # [1/min]
        
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"=== Experiment {exp_num+1} ===", True, True)
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\n== Experiment parameters ==", False, True)
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Effective time per pass: {round(time_travel,2)} [min]", False, True)
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Charge per pass: {round(charge,2)} [mA.h]", False, True)
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Machine factor (i/v): {round(ratio_iv,2)} [mA.min/m]", False, True)
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Process value (i/v/SW): {round(pv,2)} [As/m²]", False, True)
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Dose units : {units}", False, True)        
        check = np.zeros((nSource_X_cells,nSource_Y_cells)) # Used for debugging / check code
        
        # Compute one box for each object
        x_c = conv_XZ_offset[0]+mother_pallet_dim[0]+central_wooden_pallet_dim[0]+central_prod_dim[0]/2
        y_c = central_prod_yz_pos[0]
        z_c = central_prod_yz_pos[1]+central_prod_dim[2]/2
        x_l = conv_XZ_offset[0]+mother_pallet_dim[0]+left_wooden_pallet_dim[0]+left_prod_dim[0]/2
        y_l = central_prod_yz_pos[0]-left_prod_yGap-central_prod_dim[1]/2-left_prod_dim[1]/2
        z_l = central_prod_yz_pos[1]+central_prod_dim[2]/2
        x_r = conv_XZ_offset[0]+mother_pallet_dim[0]+right_wooden_pallet_dim[0]+right_prod_dim[0]/2
        y_r = central_prod_yz_pos[0]+right_prod_yGap+central_prod_dim[1]/2+right_prod_dim[1]/2
        z_r = central_prod_yz_pos[1]+central_prod_dim[2]/2
        
        Translation = np.array([x_c, y_c, z_c-central_prod_dim[2]/2])
        
        Box_central_prod = np.array([np.array([x_c,y_c,z_c]), central_prod_dim])
        Box_left_prod = np.array([np.array([x_l,y_l,z_l]), left_prod_dim])
        Box_right_prod = np.array([np.array([x_r,y_r,z_r]), right_prod_dim])
        
        Box_central_wooden_pallet = np.array([np.array([x_c-central_prod_dim[0]/2-central_wooden_pallet_dim[0]/2,y_c,z_c]), central_wooden_pallet_dim])
        Box_left_wooden_pallet = np.array([np.array([x_l-left_prod_dim[0]/2-left_wooden_pallet_dim[0]/2,y_l,z_l]), left_wooden_pallet_dim])
        Box_right_wooden_pallet = np.array([np.array([x_r-right_prod_dim[0]/2-right_wooden_pallet_dim[0]/2,y_r,z_r]), right_wooden_pallet_dim])
        
        Box_central_mother_pallet = np.array([np.array([x_c-central_prod_dim[0]/2-central_wooden_pallet_dim[0]-mother_pallet_dim[0]/2,y_c,z_c]), mother_pallet_dim])
        Box_left_mother_pallet = np.array([np.array([x_l-left_prod_dim[0]/2-left_wooden_pallet_dim[0]-mother_pallet_dim[0]/2,y_l,z_l]), mother_pallet_dim])
        Box_right_mother_pallet = np.array([np.array([x_r-right_prod_dim[0]/2-right_wooden_pallet_dim[0]-mother_pallet_dim[0]/2,y_r,z_r]), mother_pallet_dim])
        
        Box_conveyor = np.array([np.array([conv_XZ_offset[0]-conv_dim[0]/2,0,conv_XZ_offset[1]+conv_dim[2]/2]), conv_dim])
        
        # list of boxes and respective densities
        boxes = [Box_central_prod,Box_left_prod,Box_right_prod,Box_central_wooden_pallet,Box_left_wooden_pallet,Box_right_wooden_pallet,Box_central_mother_pallet,Box_left_mother_pallet,Box_right_mother_pallet,Box_conveyor]
        densities = [central_prod_density,left_prod_density,right_prod_density,central_wooden_pallet_density,left_wooden_pallet_density,right_wooden_pallet_density,mother_pallet_density,mother_pallet_density,mother_pallet_density,conv_density]
        n_boxes = len(boxes)
        
        ### loop over  X, Y, Z for 1-D mappings and save the 1D doses in all directions
        Dose_1D_single = np.empty((nXY_1D,3), dtype=object)
        Dose_1D_double = np.empty((nXY_1D,3), dtype=object)
        print("\n== Looping over 1D mappings ... ==")
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\n== 1D-mappings ==", False, False)
        
        for j in range(nXY_1D):
            save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\n= 1D-Mapping number {j} around (X,Y,Z)=({XYZ_1D[j][0]},{XYZ_1D[j][1]},{XYZ_1D[j][2]}) =", False, False)
            for i in range(3): 
                kmax = int(central_prod_dim[i]/central_prod_res[i])
                mapping = np.zeros(kmax) # 1-D array (dim = kmax) that contains the X-coordinates of the mapping
                mapping_directions = ["X","Y","Z"]
                
                Dose_single = np.zeros(kmax)
                Dose_double = np.zeros(kmax)
                
                print(f"Looping 1D-mapping number {j+1} in {mapping_directions[i]} direction ...")
                
                for k in range(kmax): # loop over mapping voxels
                    if i==2: 
                        mapping[k] = (k + 0.5)*central_prod_res[i] # [cm] - For profiles in Z, the origin in on face Z-
                    else: 
                        mapping[k] = (k + 0.5)*central_prod_res[i] - 0.5*central_prod_dim[i]  # [cm] - For profiles in X & Y, the origin in the middle of plane Z-
                    if i==0: 
                        PointInProduct_single = np.array([mapping[k]+Translation[0], XYZ_1D[j][1], XYZ_1D[j][2]])
                        PointInProduct_double = np.array([mapping[k]+Translation[0], XYZ_1D[j][1], 2*z_c-XYZ_1D[j][2]])            
                    elif i==1: 
                        PointInProduct_single = np.array([XYZ_1D[j][0], mapping[k]+Translation[1], XYZ_1D[j][2]])    
                        PointInProduct_double = np.array([XYZ_1D[j][0], mapping[k]+Translation[1], 2*z_c-XYZ_1D[j][2]])       
                    elif i==2: 
                        PointInProduct_single = np.array([XYZ_1D[j][0], XYZ_1D[j][1], mapping[k]+Translation[2]])
                        PointInProduct_double = np.array([XYZ_1D[j][0], XYZ_1D[j][1], 2*z_c-mapping[k]-Translation[2]])       

                    Dose_single[k] = compute_dose(PointInProduct_single,boxes,n_boxes,densities,nSource_X_cells,nSource_Y_cells,source_xRes,source_yRes,width_scan,length_nom,units_num,bool_current_x,bool_current_y,currents)
                    Dose_double[k] = Dose_single[k] + compute_dose(PointInProduct_double,boxes,n_boxes,densities,nSource_X_cells,nSource_Y_cells,source_xRes,source_yRes,width_scan,length_nom,units_num,bool_current_x,bool_current_y,currents)
                # Save the doses
                Dose_1D_single[j,i] = Dose_single
                Dose_1D_double[j,i] = Dose_double
                
                if i==0: 
                    if X_mapping_bool:
                        leg_single = 'Single side : X-Profile (Y = ' + str(XYZ_1D[j][1])+', Z = ' + str(XYZ_1D[j][2])+')'
                        leg_double = 'Double side : X-Profile (Y = ' + str(XYZ_1D[j][1])+', Z = ' + str(XYZ_1D[j][2])+')'
                        axs.plot(mapping+Translation[0], Dose_single,'.-', label=leg_single,
                                    color=colors[j],lw=1, alpha=1)
                        axs.plot(mapping+Translation[0], Dose_double,'.-', label=leg_double,
                                    color=colors[j],lw=2, alpha=1/2)
                elif i==1: 
                    if Y_mapping_bool:
                        leg_single = 'Single side : Y-Profile (X = ' + str(XYZ_1D[j][0])+', Z = ' + str(XYZ_1D[j][2])+')'
                        leg_double = 'Double side : Y-Profile (X = ' + str(XYZ_1D[j][0])+', Z = ' + str(XYZ_1D[j][2])+')'
                        ays.plot(mapping+Translation[1], Dose_single,'.-', label=leg_single,
                                    color=colors[j],lw=1, alpha=1)
                        ays.plot(mapping+Translation[1], Dose_double,'.-', label=leg_double,
                                    color=colors[j],lw=2, alpha=1/2)
                elif i==2: 
                    if Z_mapping_bool:
                        leg_single = 'Single side : Z-Profile (X = ' + str(XYZ_1D[j][0])+', Y = ' + str(XYZ_1D[j][1])+')'
                        leg_double = 'Double side : Z-Profile (X = ' + str(XYZ_1D[j][0])+', Y = ' + str(XYZ_1D[j][1])+')'
                        azs.plot(mapping+Translation[2], Dose_single,'.-', label=leg_single,
                                    color=colors[j],lw=1, alpha=1)
                        azs.plot(mapping+Translation[2], Dose_double,'.-', label=leg_double,
                                    color=colors[j],lw=2, alpha=1/2)
                # print and save dose
                save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\n{mapping_directions[i]} direction - single : {Dose_1D_single[j,i]} - double : {Dose_1D_double[j,i]}", False, False)
                
                
        ### loop over 0_D mapping points and save dose at those points
        print("\n== Looping over all 0-D mapping points ... ==")
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\n== 0D-mappings ==", False, False)
        Dose_0D_single = np.zeros(nXY_0D)  
        Dose_0D_double = np.zeros(nXY_0D)  
        
        for j in range(nXY_0D): 
            Translation = np.array([x_c, y_c, z_c-central_prod_dim[2]/2])
            PointInProduct_single = XYZ_0D[j]   
            PointInProduct_double = np.array([XYZ_0D[j][0],XYZ_0D[j][1],2*z_c-XYZ_0D[j][2]])
            Dose_0D_single[j] = compute_dose(PointInProduct_single,boxes,n_boxes,densities,nSource_X_cells,nSource_Y_cells,source_xRes,source_yRes,width_scan,length_nom,units_num,bool_current_x,bool_current_y,currents)
            Dose_0D_double[j] = Dose_0D_single[j] + compute_dose(PointInProduct_double,boxes,n_boxes,densities,nSource_X_cells,nSource_Y_cells,source_xRes,source_yRes,width_scan,length_nom,units_num,bool_current_x,bool_current_y,currents)
            save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\n= 0D-Mapping number {j+1} at (X,Y,Z)=({XYZ_0D[j][0]},{XYZ_0D[j][1]},{XYZ_0D[j][2]}) =", False, False)
            save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\nDose single : {Dose_0D_single[j]} {units} - dose double : {Dose_0D_double[j]} {units}", False, False)

        ### Find Dmin, Dmax and DUR for single and double side irradiation in central product
        print("\n== Looking for minimum and maximum doses in product ... ==")
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\n== Dose min, dose max and dur ==", False, False)
        func = lambda PointInProduct,boxes,n_boxes,densities,nSource_X_cells,nSource_Y_cells,source_xRes,source_yRes,width_scan,length_nom,units_num,bool_current_x,bool_current_y,currents : compute_dose(PointInProduct,boxes,n_boxes,densities,nSource_X_cells,nSource_Y_cells,source_xRes,source_yRes,width_scan,length_nom,units_num,bool_current_x,bool_current_y,currents)
        
        # Find min and max dose for single and double side irradiation
        doses_single, doses_double, doses_points = find_min_max_dose(func, central_prod_res, boxes, n_boxes, densities, nSource_X_cells, nSource_Y_cells, source_xRes, source_yRes, width_scan, length_nom, units_num, bool_current_x, bool_current_y, currents)

        # Get indices of min and max doses for single and double side irradiation
        ind_min_single = np.unravel_index(np.argmin(doses_single), doses_single.shape)
        ind_max_single = np.unravel_index(np.argmax(doses_single), doses_single.shape)
        ind_min_double = np.unravel_index(np.argmin(doses_double), doses_double.shape)
        ind_max_double = np.unravel_index(np.argmax(doses_double), doses_double.shape)

        # Retrieve dose values at the min and max indices
        dmin_single = doses_single[ind_min_single]
        dmax_single = doses_single[ind_max_single]
        dmin_double = doses_double[ind_min_double]
        dmax_double = doses_double[ind_max_double]

        # Retrieve points at the min and max indices
        dmin_point_single = doses_points[ind_min_single]
        dmin_point_double = doses_points[ind_min_double]
        dmax_point_single = doses_points[ind_max_single]
        dmax_point_double = doses_points[ind_max_double]

        # Calculate Dose Uniformity Ratio (DUR)
        dur_single = dmax_single / dmin_single
        dur_double = dmax_double / dmin_double

        # Round dose values to two decimals
        dmin_single_rounded = round(dmin_single, 2)
        dmax_single_rounded = round(dmax_single, 2)
        dmin_double_rounded = round(dmin_double, 2)
        dmax_double_rounded = round(dmax_double, 2)

        # Round the coordinates to two decimals
        dmin_point_single_rounded = [round(coord, 2) for coord in dmin_point_single]
        dmin_point_double_rounded = [round(coord, 2) for coord in dmin_point_double]
        dmax_point_single_rounded = [round(coord, 2) for coord in dmax_point_single]
        dmax_point_double_rounded = [round(coord, 2) for coord in dmax_point_double]

        # Round DUR to two decimals
        dur_single_rounded = round(dur_single, 2)
        dur_double_rounded = round(dur_double, 2)

        # Print rounded values for single side irradiation
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", "\n= Single side irradiation =", False, True)
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\nDose min : {dmin_single_rounded} {units} at point : {dmin_point_single_rounded}", False, True)
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Dose max : {dmax_single_rounded} {units} at point : {dmax_point_single_rounded}", False, True)
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"DUR : {dur_single_rounded}", False, True)

        # Print rounded values for single side irradiation
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", "\n= Double side irradiation =", False, True)
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\nDose min : {dmin_double_rounded} {units} at point : {dmin_point_double_rounded}", False, True)
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Dose max : {dmax_double_rounded} {units} at point : {dmax_point_double_rounded}", False, True)
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"DUR : {dur_double_rounded}", False, True)
                
        if units_num >= 2: 
            nLaps_single = int(minimum_dose/dmin_single)+1 # Compute nLpas based on dose
            nLaps_double = int(minimum_dose/dmin_double)+1 # Compute nLpas based on dose
        else: 
            nLaps_single = int(minimum_dose*60/time_travel/dmin_single)+1 # Compute nLpas based on dose rate
            nLaps_double = int(minimum_dose*60/time_travel/dmin_double)+1 # Compute nLpas based on dose rate
        throughput_single = nPalletPerMin/nLaps_single *60 * operation_hours # [# pallets per year]
        throughput_double = nPalletPerMin/nLaps_double *60 * operation_hours # [# pallets per year]
        
        # Save the conclusions about number of laps and throughput
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\n== Conclusions about number of laps and throughput ==", False, True)
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\nGlobal Number of laps in single side irradiation: {nLaps_single}", False, True)
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Global Number of laps in double side irradiation: {nLaps_double}", False, True)
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Throughput [Pallets/year] in single side irradiation: {round(throughput_single,1)}", False, True)
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Throughput [Pallets/year] in double side irradiation: {round(throughput_double,1)}", False, True)
        
        # Save front, middle and back dose XY plane
        plot_heatmaps(doses_single, exp_num, units, Box_central_prod[0], Box_central_prod[1], False)
        plot_heatmaps(doses_double, exp_num, units, Box_central_prod[0], Box_central_prod[1], True)
        
        ################################################   
        ####  Part 3: Format plots              ########
        ################################################                    

        if X_mapping_bool:
            legend = axs.legend(loc='best', shadow=True, fontsize=8)  
            axs.set(xlabel='X [cm]', ylabel=yaxis_title) 
            axs.set_ylim(bottom=0) #equivalent to "axs.set_ylim(ymin=0)". Note that this must be set after plotting, otherwise range is set automatically to [0 1]
            if limits[0,0] < limits[0,1]: axs.set_xlim([limits[0,0], limits[0,1]])
            if limits[0,2] < limits[0,3]: axs.set_ylim([limits[0,2], limits[0,3]])
            axs.set_title(title)
            axs.grid()
            fig_x.savefig(f"./Saved_figures/Experiment {exp_num+1} - X axis.png")
        if Y_mapping_bool:
            legend = ays.legend(loc='best', shadow=True, fontsize=8)  
            ays.set(xlabel='Y [cm]', ylabel=yaxis_title)
            ays.set_ylim(bottom=0)
            if limits[1,0] < limits[1,1]: ays.set_xlim([limits[1,0], limits[1,1]])
            if limits[1,2] < limits[1,3]: ays.set_ylim([limits[1,2], limits[1,3]])
            ays.set_title(title)
            ays.grid()
            fig_y.savefig(f"./Saved_figures/Experiment {exp_num+1} - Y axis.png")
        if Z_mapping_bool:
            legend = azs.legend(loc='best', shadow=True, fontsize=8)
            azs.set(xlabel='Z [cm]', ylabel=yaxis_title)
            azs.set_ylim(bottom=0) 
            if limits[2,0] < limits[2,1]: azs.set_xlim([limits[2,0], limits[2,1]])
            if limits[2,2] < limits[2,3]: azs.set_ylim([limits[2,2], limits[2,3]])
            azs.set_title(title)
            azs.grid()
            fig_z.savefig(f"./Saved_figures/Experiment {exp_num+1} - Z axis.png")

        # computation time
        end1 = time.time() 
        total_time = round(end1 - start,2)
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\n== Computation time for experiment ==", False, True)
        save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\nTotal computation time : {total_time} [s]", False, True)


        