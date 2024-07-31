### Python file checking the input .xlsx file ###

import matplotlib.pyplot as plt
import numpy as np
import DPfunctions as DP
import pandas as pd
import math
import re

### Set input .xlsx file ###
inputFile = 'XRayTracing_Input.xlsx'
data = pd.read_excel(inputFile, header=1)


############################################
################ Main code #################
############################################

### Check simulation information ###

# Check if 'Simulation number' column is composed of integers only and ordered
def check_integers_and_order(column):
    # Check if all values are integers
    all_integers = all(isinstance(x, int) or (isinstance(x, float) and x.is_integer()) for x in column)
    # Check if values are ordered from 1 to n
    correct_order = list(range(1, len(column) + 1))
    values_order = list(column)
    ordered = values_order == correct_order
    return all_integers and ordered

if not check_integers_and_order(data["Simulation number"]):
    print("Error in 'Simulation number' column data : at least one entry is not integer or entries are not ordered from one to the number of experiments")
    
# Check if 'Run (V or X)' column is composed of 'V' and 'X' only 
def check_V_or_X(column):
    return all(x in {'V', 'X'} for x in column)

if not check_V_or_X(data["Run (V or X)"]):
    print("Error in 'Run (V or X)' column data : at least one entry is not a 'V' or 'X'")
    
### Check source information ###

# Check X and Y source dimensions
def check_all_positive(column):
    return all(x > 0 for x in column)

if not check_all_positive(data["Source X-dimension [cm]"]):
    print("Error in 'Source X-dimension [cm]' column data : at least one entry is not positive")
    
if not check_all_positive(data["Source Y-dimension [cm]"]):
    print("Error in 'Source Y-dimension [cm]' column data : at least one entry is not positive")
    
# Check for source resolution
def is_tuple_of_two_positive_numbers(column):
    for entry in column:
        try:
            x = float(entry.split('(')[1].split(',')[0])
            y = float(entry.split(',)')[-2].split(',')[-1])
            if x<=0 or y<=0:
                return False
        except:
            return False  
    return True          

if not is_tuple_of_two_positive_numbers(data["Source resolution (X,Y,) [cm²]"]):
    print("Error in 'Source resolution (X,Y,) [cm²]' column data : at least one entry is not a tuple of two positive numbers")   

# Check for source current
if not check_all_positive(data["Current [mA]"]):
    print("Error in 'Current [mA]' column data : at least one entry is not positive")   
    
### Check for conveyor information ###

# Check for conveyor density
if not check_all_positive(data["Conveyor density [g/cm³]"]):
    print("Error in 'Conveyor density [g/cm³]' column data : at least one entry is not positive") 

# Check for conveyor positive dimensions
def is_tuple_of_three_positive_numbers(column):
    for entry in column:
        try:
            x = float(entry.split('(')[1].split(',')[0])
            y = float(entry.split(',')[1].split(',')[0])
            z = float(entry.split(')')[-2].split(',')[-1])
            if x<0 or y<0 or z<0:
                return False
        except:
            return False   
    return True

if not is_tuple_of_three_positive_numbers(data["Conveyor dimensions [cm³]"]):
    print("Error in 'Conveyor dimensions [cm³]' column data : at least one entry is not positive triplet")   

# Check for conveyor offset tuple
def is_tuple_of_two_numbers_with_last_positive(column):
    for entry in column:
        try:
            x = float(entry.split('(')[1].split(',')[0])
            y = float(entry.split(',)')[-2].split(',')[-1])
            if y<0:
                return False
        except:
            return False 
    return True

if not is_tuple_of_two_numbers_with_last_positive(data["Conveyor (X,Z,) offset to source [cm]²"]):
    print("Error in 'Conveyor (X,Z,) offset to source [cm]²' column data : at least one entry is not a tuple of two numbers")   

# Check for conveyor speed
if not check_all_positive(data["Conveyor speed [m/min]"]):
    print("Error in 'Conveyor speed [m/min]' column data : at least one entry is not positive")   

# Check for operation hours
if not check_all_positive(data["Operation [hours/year]"]):
    print("Error in 'Operation [hours/year]' column data : at least one entry is not positive")   
    
### Check for central product information ###

# Check for central product density
if not check_all_positive(data["Central product density [g/cm³]"]):
    print("Error in 'Central product density [g/cm³]' column data : at least one entry is not positive")
    
# Check for central product dimensions
if not is_tuple_of_three_positive_numbers(data["Central product dimensions [cm³]"]):
    print("Error in 'Central product dimensions [cm³]' column data : at least one entry is not positive")
    
# Check for central product (Y,Z) position
if not is_tuple_of_two_numbers_with_last_positive(data["Central product (Y,Z,) position [cm]"]):
    print("Error in 'Central product (Y,Z) position [cm]' column data : at least one entry is not a tuple of two positive numbers")

# Check for central product resolution      
if not is_tuple_of_three_positive_numbers(data["Central product resolution (X,Y,Z) [cm³]"]):
    print("Error in 'Central product resolution (X,Y,Z) [cm³]' column data : at least one entry is not a tuple of three positive numbers")

### check for central wooden pallet information ###

# check for central wooden pallet density
if not check_all_positive(data["Central wooden pallet density [g/cm³]"]):
    print("Error in 'Central wooden pallet density [g/cm³]' column data : at least one entry is not positive")

# Check for central wooden pallet dimensions
if not is_tuple_of_three_positive_numbers(data["Central wooden pallet dimensions [cm³]"]):
    print("Error in 'Central wooden pallet dimensions [cm³]' column data : at least one entry is not a tuple of three positive numbers")

### Check for left product information ###

# Check for left product density
if not check_all_positive(data["Left product density [g/cm³]"]):
    print("Error in 'Left product density [g/cm³]' column data : at least one entry is not positive")
    
# Check for left product dimensions
if not is_tuple_of_three_positive_numbers(data["Left product dimensions [cm³]"]):
    print("Error in 'Left product dimensions [cm³]' column data : at least one entry is not positive")
    
# Check for left product Y-gap positivity
if not check_all_positive(data["Left product Y-gap [cm]"]):
    print("Error in 'Left product Y-gap [cm]' column data : at least one entry is not positive")

### check for left wooden pallet information ###

# check for central wooden pallet density
if not check_all_positive(data["Left wooden pallet density [g/cm³]"]):
    print("Error in 'Left wooden pallet density [g/cm³]' column data : at least one entry is not positive")

# Check for left wooden pallet dimensions
if not is_tuple_of_three_positive_numbers(data["Left wooden pallet dimensions [cm³]"]):
    print("Error in 'Left wooden pallet dimensions [cm³]' column data : at least one entry is not a tuple of three positive numbers")


### Check for right product information ###

# Check for right product density
if not check_all_positive(data["Right product density [g/cm³]"]):
    print("Error in 'Right product density [g/cm³]' column data : at least one entry is not positive")
    
# Check for right product dimensions
if not is_tuple_of_three_positive_numbers(data["Right product dimensions [cm³]"]):
    print("Error in 'Right product dimensions [cm³]' column data : at least one entry is not positive")
    
# Check for right product Y-gap positivity
if not check_all_positive(data["Right product Y-gap [cm]"]):
    print("Error in 'right product Y-gap [cm]' column data : at least one entry is not positive")

### check for right wooden pallet information ###

# check for right wooden pallet density
if not check_all_positive(data["Right wooden pallet density [g/cm³]"]):
    print("Error in 'Right wooden pallet density [g/cm³]' column data : at least one entry is not positive")

# Check for right wooden pallet dimensions
if not is_tuple_of_three_positive_numbers(data["Right wooden pallet dimensions [cm³]"]):
    print("Error in 'Right wooden pallet dimensions [cm³]' column data : at least one entry is not a tuple of three positive numbers")

### Check for mother pallet information ###

# Check for mother pallet density
if not check_all_positive(data["Mother pallets density [g/cm³]"]):
    print("Error in 'Mother pallets density [g/cm³]' column data : at least one entry is not positive")

# Check for mother pallet dimensions
if not is_tuple_of_three_positive_numbers(data["Mother pallets dimensions [cm³]"]):
    print("Error in 'Mother pallets dimensions [cm³]' column data : at least one entry is not a tuple of three positive numbers")

### Check for dose information ###
def check_integer_between_0_3(column):
    for value in column:
        if not isinstance(value, int) or not (0 <= value <= 3):
            return False
    return True

# Check for dose units
if not check_integer_between_0_3(data["Units (Gy/h) (kGy/h) (Gy) (kGy)"]):
    print("Error in 'Units (Gy/h) (kGy/h) (Gy) (kGy)' column data : at least one entry is not an integer between 0 and 3")

# check for minimum dose
if not check_all_positive(data["Minimum dose"]):
    print("Error in 'Minimum dose' column data : at least one entry is not positive")

### Check for 0-D mapping points ###

# Check for list of 0-D mappings
def validate_tuple_string(s):
    if s == "" or (isinstance(s, float) and math.isnan(s)):
        return True

    pattern = r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)'

    # Find all tuples in the string
    matches = re.findall(pattern, s)

    # Check if each match contains positive integers
    for match in matches:
        x, y, z = map(float, match)

    # Return True if all tuples are valid
    return len(matches) > 0 and len(matches) == s.count("(")

def is_list_of_triplets(column):
    for entry in column:
        if not validate_tuple_string(entry):
            return False
    return True 

if not is_list_of_triplets(data["List of n 0-D dose mapping points (X1,Y1,Z1), …, (Xn,Yn,Zn)"]):
    print("Error in 'List of n 0-D dose mapping points (X1,Y1,Z1), …, (Xn,Yn,Zn)' column data : at least one entry is not a list of triplets")

### Check for 1-D mapping points ###

# Check for X-mapping (V or X)
if not check_V_or_X(data["X-mapping (V or X)"]):
    print("Error in 'X-mapping (V or X)' column data : at least one entry is not 'V' or 'X")
    
# Check for Y-mapping (V or X)
if not check_V_or_X(data["Y-mapping (V or X)"]):
    print("Error in 'Y-mapping (V or X)' column data : at least one entry is not 'V' or 'X")
    
# Check for Z-mapping (V or X)
if not check_V_or_X(data["Z-mapping (V or X)"]):
    print("Error in 'Z-mapping (V or X)' column data : at least one entry is not 'V' or 'X")
    
# Check for list of 1-D mappings
if not is_list_of_triplets(data["List of p 1-D mapping points (X1,Y1,Z1), …, (Xp,Yp,Zp)"]):
    print("Error in 'List of p 1-D mapping points (X1,Y1,Z1), …, (Xp,Yp,Zp)' column data : at least one entry is not a list of triplets")

### Check for overlaps

# Check for left and right products source collision
bool_print_left = False
bool_print_right = False
if is_tuple_of_three_positive_numbers(data["Left product dimensions [cm³]"]):
    if is_tuple_of_three_positive_numbers(data["Right product dimensions [cm³]"]):
        if is_tuple_of_three_positive_numbers(data["Central product dimensions [cm³]"]):
            if is_tuple_of_two_numbers_with_last_positive(data["Central product (Y,Z,) position [cm]"]):
                for index,row in data.iterrows():
                    entry_c1 = row["Central product dimensions [cm³]"]
                    entry_c2 = row["Central product (Y,Z,) position [cm]"]
                    entry_l = row["Left product dimensions [cm³]"]
                    entry_r = row["Right product dimensions [cm³]"]
                   
                    Z_c = float(entry_c1.split(')')[-2].split(',')[-1])
                    z_c = float(entry_c2.split(',)')[-2].split(',')[-1])
                    Z_l = float(entry_l.split(')')[-2].split(',')[-1])
                    Z_r = float(entry_r.split(')')[-2].split(',')[-1])
                    if z_c+Z_c/2-Z_l/2 <= 0:
                        bool_print_left = True
                    if z_c+Z_c/2-Z_r/2 <= 0:
                        bool_print_right = True

if bool_print_left:
    print("Error : based on left and central product informations, at least one left product will overlap the source in z=0 plane")
if bool_print_right:
    print("Error : based on right and central product informations, at least one right product will overlap the source in z=0 plane")

# Check for pallets collisions in y and z directions
bool_print_left_y = False
bool_print_left_z = False
bool_print_right_y = False
bool_print_right_z = False
bool_print_central_z = False
if is_tuple_of_three_positive_numbers(data["Left product dimensions [cm³]"]):
    if is_tuple_of_three_positive_numbers(data["Right product dimensions [cm³]"]):
        if is_tuple_of_three_positive_numbers(data["Central product dimensions [cm³]"]):
            if is_tuple_of_two_numbers_with_last_positive(data["Central product (Y,Z,) position [cm]"]):
                if check_all_positive(data["Left product Y-gap [cm]"]) and check_all_positive(data["Right product Y-gap [cm]"]):
                    if is_tuple_of_three_positive_numbers(data["Left wooden pallet dimensions [cm³]"]):
                        if is_tuple_of_three_positive_numbers(data["Right wooden pallet dimensions [cm³]"]):
                            if is_tuple_of_three_positive_numbers(data["Mother pallets dimensions [cm³]"]):
                                for index,row in data.iterrows():
                                    entry_c1 = row["Central product dimensions [cm³]"]
                                    entry_c2 = row["Central product (Y,Z,) position [cm]"]
                                    entry_l = row["Left product dimensions [cm³]"]
                                    entry_r = row["Right product dimensions [cm³]"]
                                    entry_gl = row["Left product Y-gap [cm]"]
                                    entry_gr = row["Right product Y-gap [cm]"]
                                    entry_pl = row["Left wooden pallet dimensions [cm³]"]
                                    entry_pr = row["Right wooden pallet dimensions [cm³]"]
                                    entry_pc = row["Central wooden pallet dimensions [cm³]"]
                                    entry_mp = row["Mother pallets dimensions [cm³]"]
                                    # z dimensions
                                    Z_c = float(entry_c1.split(')')[-2].split(',')[-1])
                                    z_c = float(entry_c2.split(',)')[-2].split(',')[-1])
                                    Z_l = float(entry_l.split(')')[-2].split(',')[-1])
                                    Z_r = float(entry_r.split(')')[-2].split(',')[-1])
                                    Z_pl = float(entry_pl.split(')')[-2].split(',')[-1])
                                    Z_pr = float(entry_pr.split(')')[-2].split(',')[-1])
                                    Z_pc = float(entry_pc.split(')')[-2].split(',')[-1])
                                    Z_mp = float(entry_mp.split(')')[-2].split(',')[-1])
                                    # y dimensions
                                    Y_c = float(entry_c1.split(',')[1].split(',')[0])
                                    y_c = float(entry_c2.split('(')[1].split(',')[0])
                                    Y_l = float(entry_l.split(',')[1].split(',')[0])
                                    Y_r = float(entry_r.split(',')[1].split(',')[0])
                                    Y_pl = float(entry_pl.split(',')[1].split(',')[0])
                                    Y_pr = float(entry_pr.split(',')[1].split(',')[0])
                                    Y_pc = float(entry_pc.split(',')[1].split(',')[0])
                                    Y_mp = float(entry_mp.split(',')[1].split(',')[0])
                                    # gaps
                                    gap_l = float(entry_gl)
                                    gap_r = float(entry_gr)
                                    # check collisions Z
                                    if z_c+Z_c/2-Z_pl/2 <= 0 or z_c+Z_c/2-Z_mp/2 <= 0:
                                        bool_print_left_z = True
                                    if z_c+Z_c/2-Z_pr/2 <= 0 or z_c+Z_c/2-Z_mp/2 <= 0:
                                        bool_print_right_z = True
                                    if z_c+Z_c/2-Z_pc/2 <= 0 or z_c+Z_c/2-Z_mp/2 <= 0:
                                        bool_print_central_z = True
                                    # check collisions Y
                                    if -Y_pc/2<= -Y_c/2-gap_l-Y_l/2+Y_pl/2 or -Y_mp/2<= -Y_c/2-gap_l-Y_l/2+Y_mp/2:
                                        bool_print_left_y = True
                                    if Y_pc/2>= Y_c/2+gap_r+Y_r/2-Y_pr/2 or Y_mp/2>= Y_c/2+gap_r+Y_r/2-Y_mp/2:
                                        bool_print_right_y = True
if bool_print_left_y:
    print("Error : based on left and central product and pallets informations, at least one left wooden/mother pallet will overlap the central wooden/mother pallet")
if bool_print_right_y:
    print("Error : based on right and central product and pallets informations, at least one right wooden/mother pallet will overlap the central wooden/mother pallet")
if bool_print_central_z:
    print("Error : based on central product and pallets informations, at least one central wooden/mother pallet will overlap the source in z=0 plane")
if bool_print_right_z:
    print("Error : based on right and central product/pallets informations, at least one right wooden/mother pallet will overlap the source in z=0 plane")
if bool_print_left_z:
    print("Error : based on left and central product/pallets informations, at least one left wooden/mother pallet will overlap the source in z=0 plane")

### Check for points in central product

# Check for 0-D and 1-D mappings
if is_list_of_triplets(data["List of n 0-D dose mapping points (X1,Y1,Z1), …, (Xn,Yn,Zn)"]):
    if is_list_of_triplets(data["List of p 1-D mapping points (X1,Y1,Z1), …, (Xp,Yp,Zp)"]):
        if is_tuple_of_three_positive_numbers(data["Central product dimensions [cm³]"]):
            if is_tuple_of_two_numbers_with_last_positive(data["Central product (Y,Z,) position [cm]"]):
                if is_tuple_of_three_positive_numbers(data["Conveyor dimensions [cm³]"]):
                    if is_tuple_of_two_numbers_with_last_positive(data["Conveyor (X,Z,) offset to source [cm]²"]):
                        if is_tuple_of_three_positive_numbers(data["Central wooden pallet dimensions [cm³]"]):
                            if is_tuple_of_three_positive_numbers(data["Mother pallets dimensions [cm³]"]):
                                for index,row in data.iterrows():
                                    entry_0D = row["List of n 0-D dose mapping points (X1,Y1,Z1), …, (Xn,Yn,Zn)"]
                                    entry_1D = row["List of p 1-D mapping points (X1,Y1,Z1), …, (Xp,Yp,Zp)"]
                                    entry_c1 = row["Central product dimensions [cm³]"]
                                    entry_c2 = row["Central product (Y,Z,) position [cm]"]
                                    entry_conv1 = row["Conveyor dimensions [cm³]"]
                                    entry_conv2 = row["Conveyor (X,Z,) offset to source [cm]²"]
                                    entry_pc = row["Central wooden pallet dimensions [cm³]"]
                                    entry_mp = row["Mother pallets dimensions [cm³]"]
                                    
                                    X_c = float(entry_c1.split('(')[1].split(',')[0])
                                    Y_c = float(entry_c1.split(',')[1].split(',')[0])
                                    Z_c = float(entry_c1.split(')')[-2].split(',')[-1])
                                    
                                    y_c = float(entry_c2.split('(')[1].split(',')[0])
                                    z_c = float(entry_c2.split(',)')[-2].split(',')[-1])
                                    x_conv = float(entry_conv2.split('(')[1].split(',')[0])
                                    #X_conv = float(entry_conv1.split('(')[1].split(',')[0])
                                    X_pc = float(entry_pc.split('(')[1].split(',')[0])
                                    X_mp = float(entry_mp.split('(')[1].split(',')[0])
                                    x_c = x_conv + X_mp + X_pc + X_c/2
                                    
                                    # check 0-D mapping points
                                    if not (entry_0D == "" or (isinstance(entry_0D, float) and math.isnan(entry_0D))):
                                        pattern = r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)'
                                        # Find all tuples in the string
                                        matches = re.findall(pattern, entry_0D)
                                        # Check each match
                                        for match in matches:
                                            x, y, z = map(float, match)
                                            if (x < x_c - X_c/2) or (x > x_c + X_c/2) or (y < y_c - Y_c/2) or (y > y_c + Y_c/2) or (z < z_c) or (z > z_c + Z_c):
                                                print(x,x_c,X_c)
                                                print(f"In 0-D mapping points, point ({x},{y},{z}) is not contained in the central product")
                                    
                                    # check 1-D mapping points
                                    if not (entry_1D == "" or (isinstance(entry_1D, float) and math.isnan(entry_1D))):
                                        pattern = r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)'
                                        # Find all tuples in the string
                                        matches = re.findall(pattern, entry_1D)
                                        # Check each match
                                        for match in matches:
                                            x, y, z = map(float, match)
                                            if (x < x_c - X_c/2) or (x > x_c + X_c/2) or (y < y_c - Y_c/2) or (y > y_c + Y_c/2) or (z < z_c) or (z > z_c + Z_c):
                                                print(f"In 1-D mapping points, point ({x},{y},{z}) is not contained in the central product")

# If no error message                                          
print("If no error messages, then the excel file was successfully completed !")