# -*- coding: utf-8 -*-
"""
XRayTracing_script.py

Created on Fri Jul 19 2024

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
import GFfunctions as GF
import pandas as pd
import math
import sys
import time
import re

### Choose the pathname and files
pathname = "" #r'C:\Users\dp\OneDrive - IBA Group\My Documents - Operationnel\1. My Projects\Y21 RayXpert\Y22_03 - XR Source Definition'
inputFile = 'XRayTracing_Input.xlsx' #pathname + '\\' + 'XRayTracing.inp'


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
        
        method = data.loc[exp_num, "Method (RBF, Linear, Angle2Dose)"].strip()
        
        # 0.1 - Read source info
        
        Energy_list = [7] # To complete if new energies
        energy = float(data.loc[exp_num, "Energy [MeV]"])
        if energy not in Energy_list:
            print("Error: {energy}[MeV] Energy not set up is XRayTracing Script")
            continue
            
        length_nom = float(data.loc[exp_num, "Source Y-dimension [cm]"])

        width_scan = float(data.loc[exp_num, "Source X-dimension [cm]"])

        source_xRes,source_yRes = [float(data.loc[exp_num, "Source resolution (X,Y,) [cm²]"].split("(")[1].split(",")[0]),
                                   float(data.loc[exp_num, "Source resolution (X,Y,) [cm²]"].split(",")[1].split(",")[0])]
        
        current = float(data.loc[exp_num, "Current [mA]"])
        
        bool_variable = data.loc[exp_num, "Variable Scan (V or X)"] == "V"
        variable_scan_path = data.loc[exp_num, "Variable Scan File Path"].strip()
        alphas, positions, lengths = GF.parse_input_file(variable_scan_path)
        if positions[0][0] == 0:
            x_variable_bool = False
        else:
            x_variable_bool = True
        source_cell_coord, source_cell_weights = GF.compute_source_cell_info(bool_variable, x_variable_bool, source_xRes, source_yRes, width_scan, length_nom, positions, lengths, alphas)

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
        minimum_dose = float(data.loc[exp_num, "Minimum dose [kGy]"])
        
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
        
        # 0.13 - Read Dur search
        bool_dur_3_planes = data.loc[exp_num, "3 XY-planes (V or X)"] == "V"
        bool_dur_9_lines = data.loc[exp_num, "9 vertical 1D lines (V or X)"] == "V"
        bool_dur_6_lines = data.loc[exp_num, "6 vertical 1D lines (V or X)"] == "V"
        bool_no_dur = data.loc[exp_num, "No search (V or X)"] == "V"

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
        time_travel = 0.01*length_nom/conv_speed # [min] = Time for the pallet to go along the entire source (>< process time!)
        charge = current*time_travel/60 # [mA h]
        ratio_iv = current/conv_speed # [mA min/m]
        pv = 6*ratio_iv/width_scan # process value [As/m²]
        nPalletPerMin = 100*conv_speed/(central_prod_dim[1] + left_prod_yGap) # [1/min]
        
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"=== Experiment {exp_num+1} ===", True, True)
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\n== Experiment parameters ==", False, True)
        
        if bool_variable and x_variable_bool: 
            GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Variable Scan mode in X-direction", False, True)
        elif bool_variable and not x_variable_bool: 
            GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Variable Scan mode in Y-direction", False, True)
        else: 
            GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Homogeneous Scan mode", False, True)
        
        if bool_no_dur: 
            GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"No Dur Computation", False, True)
        elif bool_dur_9_lines: 
            GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Dur Computation based on 9 vertical line mappings", False, True)
        elif bool_dur_3_planes: 
            GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Dur Computation based on 3 XY-planes", False, True)
        elif bool_dur_6_lines: 
            GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Dur Computation based on 6 vertical line mappings", False, True)
        
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Kernel Method : {method}", False, True)         
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Effective time per pass: {round(time_travel,2)} [min]", False, True)
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Charge per pass: {round(charge,2)} [mA.h]", False, True)
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Machine factor (i/v): {round(ratio_iv,2)} [mA.min/m]", False, True)
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Process value (i/v/SW): {round(pv,2)} [As/m²]", False, True)
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Dose units : {units}", False, True)        

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
        boxes_temp = [Box_central_prod,Box_left_prod,Box_right_prod,Box_central_wooden_pallet,Box_left_wooden_pallet,Box_right_wooden_pallet,Box_central_mother_pallet,Box_left_mother_pallet,Box_right_mother_pallet,Box_conveyor]
        densities_temp = [central_prod_density,left_prod_density,right_prod_density,central_wooden_pallet_density,left_wooden_pallet_density,right_wooden_pallet_density,mother_pallet_density,mother_pallet_density,mother_pallet_density,conv_density]
        boxes = []
        densities = []
        for i,box in enumerate(boxes_temp):
            if densities_temp[i] > 0:
                boxes.append(box)
                densities.append(densities_temp[i])
        n_boxes = len(boxes)
        
        ### loop over  X, Y, Z for 1-D mappings and save the 1D doses in all directions
        Dose_1D_single = np.empty((nXY_1D,3), dtype=object)
        Dose_1D_double = np.empty((nXY_1D,3), dtype=object)
        print("\n== Looping over 1D mappings ... ==")
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\n== 1D-mappings ==", False, False)
        
        for j in range(nXY_1D):
            GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\n= 1D-Mapping number {j} around (X,Y,Z)=({XYZ_1D[j][0]},{XYZ_1D[j][1]},{XYZ_1D[j][2]}) =", False, False)
            for i in range(3): 
                kmax = int(central_prod_dim[i]/central_prod_res[i])
                mapping = np.zeros(kmax) # 1-D array (dim = kmax) that contains the X-coordinates of the mapping
                mapping_directions = ["X","Y","Z"]
                
                if (i == 0 and not X_mapping_bool) or (i == 1 and not Y_mapping_bool) or (i == 2 and not Z_mapping_bool):
                    break
                
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

                    Dose_single[k] = GF.compute_dose(energy, method,PointInProduct_single,boxes,n_boxes,densities,source_cell_coord,source_cell_weights,units_num,current,charge)
                    Dose_double[k] = Dose_single[k] + GF.compute_dose(energy, method,PointInProduct_double,boxes,n_boxes,densities,source_cell_coord,source_cell_weights,units_num,current,charge)
                # Save the doses
                Dose_1D_single[j,i] = Dose_single
                Dose_1D_double[j,i] = Dose_double
                
                if i==0: 
                    if X_mapping_bool:
                        leg_single = 'Single side : X-Profile (Y = ' + str(XYZ_1D[j][1])+', Z = ' + str(XYZ_1D[j][2])+')'
                        leg_double = 'Double side : X-Profile (Y = ' + str(XYZ_1D[j][1])+', Z = ' + str(XYZ_1D[j][2])+')'
                        #axs.plot(mapping+Translation[0], Dose_single,'.-', label=leg_single,
                                    #color=colors[j],lw=1, alpha=1)
                        axs.plot(mapping+Translation[0], Dose_double,'.-', label=leg_double,
                                    color=colors[j],lw=2, alpha=1/2)
                        print(mapping+Translation[0])
                        print(Dose_double)
                elif i==1: 
                    if Y_mapping_bool:
                        leg_single = 'Single side : Y-Profile (X = ' + str(XYZ_1D[j][0])+', Z = ' + str(XYZ_1D[j][2])+')'
                        leg_double = 'Double side : Y-Profile (X = ' + str(XYZ_1D[j][0])+', Z = ' + str(XYZ_1D[j][2])+')'
                        #ays.plot(mapping+Translation[1], Dose_single,'.-', label=leg_single,
                                    #color=colors[j],lw=1, alpha=1)
                        ays.plot(mapping+Translation[1], Dose_double,'.-', label=leg_double,
                                    color=colors[j],lw=2, alpha=1/2)
                        print(mapping+Translation[1])
                        print(Dose_double)
                elif i==2: 
                    if Z_mapping_bool:
                        leg_single = 'Single side : Z-Profile (X = ' + str(XYZ_1D[j][0])+', Y = ' + str(XYZ_1D[j][1])+')'
                        leg_double = 'Double side : Z-Profile (X = ' + str(XYZ_1D[j][0])+', Y = ' + str(XYZ_1D[j][1])+')'
                        #azs.plot(mapping+Translation[2], Dose_single,'.-', label=leg_single,
                                    #color=colors[j],lw=1, alpha=1)
                        azs.plot(mapping+Translation[2], Dose_double,'.-', label=leg_double,
                                    color=colors[j],lw=2, alpha=1/2)
                        print(mapping+Translation[2])
                        print(Dose_double)
                # print and save dose
                GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\n{mapping_directions[i]} direction - single : {Dose_1D_single[j,i]} - double : {Dose_1D_double[j,i]}", False, False)
                
        ### loop over 0_D mapping points and save dose at those points
        print("\n== Looping over all 0-D mapping points ... ==")
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\n== 0D-mappings ==", False, False)
        Dose_0D_single = np.zeros(nXY_0D)  
        Dose_0D_double = np.zeros(nXY_0D)  
        
        for j in range(nXY_0D): 
            Translation = np.array([x_c, y_c, z_c-central_prod_dim[2]/2])
            PointInProduct_single = XYZ_0D[j]   
            PointInProduct_double = np.array([XYZ_0D[j][0],XYZ_0D[j][1],2*z_c-XYZ_0D[j][2]])
            Dose_0D_single[j] = GF.compute_dose(energy, method,PointInProduct_single,boxes,n_boxes,densities,source_cell_coord,source_cell_weights,units_num,current,charge)
            Dose_0D_double[j] = Dose_0D_single[j] + GF.compute_dose(energy, method,PointInProduct_double,boxes,n_boxes,densities,source_cell_coord,source_cell_weights,units_num,current,charge)
            GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\n= 0D-Mapping number {j+1} at (X,Y,Z)=({XYZ_0D[j][0]},{XYZ_0D[j][1]},{XYZ_0D[j][2]}) =", False, False)
            GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\nDose single : {Dose_0D_single[j]} {units} - dose double : {Dose_0D_double[j]} {units}", False, False)

        ### Find Dmin, Dmax and DUR for single and double side irradiation in central product
        print("\n== Looking for minimum and maximum doses in product ... ==")
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\n== Dose min, dose max and dur ==", False, False)
        func = lambda PointInProduct,boxes,n_boxes,densities,source_cell_coord,source_cell_weights,units_num,current,charge : GF.compute_dose(energy, method,PointInProduct,boxes,n_boxes,densities,source_cell_coord,source_cell_weights,units_num,current,charge)
        
        # Find min and max dose for single and double side irradiation
        if bool_dur_3_planes:
            doses_single, doses_double, doses_points = GF.find_min_max_dose_3planes(func, central_prod_res, boxes,n_boxes,densities,source_cell_coord,source_cell_weights,units_num,current,charge)
        elif bool_dur_9_lines:
            doses_single, doses_double, doses_points = GF.find_min_max_dose_9lines(func, central_prod_res, boxes,n_boxes,densities,source_cell_coord,source_cell_weights,units_num,current,charge)
        elif bool_dur_6_lines:
            doses_single, doses_double, doses_points = GF.find_min_max_dose_6lines(func, central_prod_res, boxes,n_boxes,densities,source_cell_coord,source_cell_weights,units_num,current,charge)  
        elif bool_no_dur:
            doses_single, doses_double, doses_points = np.array([np.inf]),np.array([np.inf]),np.array([[np.inf,np.inf,np.inf]])
            
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
        if bool_no_dur:
            dur_single = None
            dur_double = None
        else:
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
        if bool_no_dur:
            dur_single_rounded = None
            dur_double_rounded = None
        else:
            dur_single_rounded = round(dur_single, 2)
            dur_double_rounded = round(dur_double, 2)

        # Print rounded values for single side irradiation
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", "\n= Single side irradiation =", False, True)
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\nDose min : {dmin_single_rounded} {units} at point : {dmin_point_single_rounded}", False, True)
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Dose max : {dmax_single_rounded} {units} at point : {dmax_point_single_rounded}", False, True)
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"DUR : {dur_single_rounded}", False, True)

        # Print rounded values for single side irradiation
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", "\n= Double side irradiation =", False, True)
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\nDose min : {dmin_double_rounded} {units} at point : {dmin_point_double_rounded}", False, True)
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Dose max : {dmax_double_rounded} {units} at point : {dmax_point_double_rounded}", False, True)
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"DUR : {dur_double_rounded}", False, True)
        
        if not bool_no_dur:      
            if units_num >= 2: 
                nLaps_single = int(minimum_dose/dmin_single)+1 # Compute nLpas based on dose
                nLaps_double = int(minimum_dose/dmin_double)+1 # Compute nLpas based on dose
            else: 
                nLaps_single = int(minimum_dose*60/time_travel/dmin_single)+1 # Compute nLpas based on dose rate
                nLaps_double = int(minimum_dose*60/time_travel/dmin_double)+1 # Compute nLpas based on dose rate
            throughput_single = nPalletPerMin/nLaps_single *60 * operation_hours # [# pallets per year]
            throughput_double = nPalletPerMin/nLaps_double *60 * operation_hours # [# pallets per year]
        else:
            nLaps_single = None
            nLaps_double = None
            throughput_single = None
            throughput_double = None
            
        # Save the conclusions about number of laps and throughput
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\n== Conclusions about number of laps and throughput ==", False, True)
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\nGlobal Number of laps in single side irradiation: {nLaps_single}", False, True)
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Global Number of laps in double side irradiation: {nLaps_double}", False, True)
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Throughput [Pallets/year] in single side irradiation: {throughput_single}", False, True)
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"Throughput [Pallets/year] in double side irradiation: {throughput_double}", False, True)
        
        # Save front, middle and back dose XY plane
        if bool_dur_3_planes:
            GF.plot_heatmaps(doses_single, exp_num, units, Box_central_prod[0], Box_central_prod[1], False)
            GF.plot_heatmaps(doses_double, exp_num, units, Box_central_prod[0], Box_central_prod[1], True)
        
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
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\n== Computation time for experiment ==", False, True)
        GF.save_log(f"./Log_files/Exp{exp_num+1}_log_file.txt", f"\nTotal computation time : {total_time} [s]", False, True)


        