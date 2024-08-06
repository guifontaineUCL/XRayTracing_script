# XRayTracing Script - script version 1.0 - README

---

## Overview

The `XRayTracing_script.py` is a Python script designed to simulate the dose distribution of X-ray radiation on a central product from a radiation source. The simulation considers both source-related parameters (such as energy-angle spectrum and geometry) and system parameters (including geometry and density) for the conveyor, as well as the left, right, and central products, wooden pallets, and mother pallets.

The principal objective of this script is to provide a faster alternative to Monte Carlo simulations, such as those performed by the RayXpert tool, which are often too slow for practical use. This script aims to achieve an optimal balance between speed and accuracy.

The simulation has three primary objectives:
1. **Compute 1D Dose Mappings:** Calculate and plot dose profiles along the X, Y, and Z directions around specific 1D mapping points.
2. **Compute 0D Dose Mappings:** Determine the dose at discrete X, Y, Z points (0-D mappings).
3. **Extract Dose Metrics:** Evaluate the minimum dose, maximum dose, and dose uniformity ratio (DUR) within the central product for both single and double irradiation scenarios.

The script is designed to handle multiple experimental setups, which can be specified in the `XRayTracing_Input.xlsx` file.

## Files and Dependencies

1. **Input File: `XRayTracing_Input.xlsx`**
   - This Excel file contains all the parameters required for the simulation.
   - Each row corresponds to a different experiment.
   - Each column represents a specific parameter (e.g., source details, material properties, geometry of products and pallets).

2. **Checking Script: `Checking_input.py`**
   - This Python script verifies the integrity and consistency of the data provided in the input file.
   - It checks for missing or invalid entries and ensures all necessary parameters are correctly filled before running the simulation.

3. **Main Script: `XRayTracing_script.py` (Dependencies : DPfunctions.py)**
   - The primary script that performs the X-ray dose simulation.
   - It reads the input data, processes the energy-angle spectrum, and simulates the dose received by the central product. The simulation process is based on the original XRayTracing.py script written by Damien Prieels.
   - It models interactions of X-rays with the conveyor system, products, pallets, and other materials to produce accurate dose mappings. 
   - The results are saved in './Log_files/' and './Saved_figures/' directories.

## Output Files and Results

The results are saved in two ways:
1. **Log Files:**
   - Located in the `./Log_files/` directory.
   - For each experiment, the 0D and 1D mappings are saved, along with the minimum dose, maximum dose, and DUR for both single and double side irradiation scenarios. Additional relevant quantities are also computed and saved.

2. **Saved Figures:**
   - Located in the `./Saved_figures/` directory.
   - Contains plots of the 1D dose mappings along chosen X, Y, and Z directions.
   - Includes dose distribution plots in the front, middle, and back XY-planes of the central product, which are used to determine the minimum and maximum doses.


## Workflow

1. **Prepare the Input Data:**
   - Fill in the required experimental parameters in the `XRayTracing_Input.xlsx` file.
   - Ensure that each row corresponds to a different experiment and each column is filled with the appropriate data.

2. **Run Data Checking:**
   - Execute the `Checking_input.py` script to validate the input data.
   - Address any issues or errors flagged by this script before proceeding to the next step.

3. **Run the Simulation:**
   - Execute the `XRayTracing_script.py` script.
   - The script will process the input data and perform the dose simulation.
   - Results will be saved automatically in `./Log_files/` and `./Saved_figures/` directories.

4. **Analyze Results:**
   - Review the generated dose mappings and durs.
   - These results can be used to assess the radiation exposure and ensure safety or effectiveness for the products under test.

## Key Parameters in `XRayTracing_Input.xlsx`

The `XRayTracing_Input.xlsx` file is organized into several sections, each containing specific parameters needed for the X-ray dose simulation. Below is a breakdown of the key parameters grouped by their respective sections:

### Simulation Information
- **Simulation number:** Unique identifier for each experiment.
- **Run (V or X):** Indicates whether the simulation should be executed (V) or skipped (X).

### Source Information
- **Source X-dimension [cm]:** Width of the X-ray source.
- **Source Y-dimension [cm]:** Length of the X-ray source.
- **Source resolution (X, Y) [cm²]:** Resolution of the source grid in the X and Y directions.
- **Current [mA]:** Current of the X-ray source, determining its intensity.

### Conveyor Information
- **Conveyor density [g/cm³]:** Material density of the conveyor.
- **Conveyor dimensions [cm³]:** Length, width, and height of the conveyor.
- **Conveyor (X, Z) offset to source [cm]:** Horizontal and vertical offset of the conveyor relative to the source.
- **Conveyor speed [m/min]:** Speed of the conveyor belt.
- **Operation [hours/year]:** Total operational hours per year.

### Central Product Information
- **Central product density [g/cm³]:** Material density of the central product.
- **Central product dimensions [cm³]:** Length, width, and height of the central product.
- **Central product (Y, Z) position [cm]:** Position of the central product in Y and Z coordinates relative to the conveyor.
- **Central product resolution (X, Y, Z) [cm³]:** Grid resolution for the central product in the X, Y, and Z directions.

### Central Wooden Pallet Information
- **Central wooden pallet density [g/cm³]:** Material density of the wooden pallet beneath the central product.
- **Central wooden pallet dimensions [cm³]:** Length, width, and height of the central wooden pallet.

### Left Product Information
- **Left product density [g/cm³]:** Material density of the product on the left side of the conveyor.
- **Left product dimensions [cm³]:** Length, width, and height of the left product.
- **Left product Y-gap [cm]:** Gap between the left product and the central product in the Y direction.

### Left Wooden Pallet Information
- **Left wooden pallet density [g/cm³]:** Material density of the wooden pallet beneath the left product.
- **Left wooden pallet dimensions [cm³]:** Length, width, and height of the left wooden pallet.

### Right Product Information
- **Right product density [g/cm³]:** Material density of the product on the right side of the conveyor.
- **Right product dimensions [cm³]:** Length, width, and height of the right product.
- **Right product Y-gap [cm]:** Gap between the right product and the central product in the Y direction.

### Right Wooden Pallet Information
- **Right wooden pallet density [g/cm³]:** Material density of the wooden pallet beneath the right product.
- **Right wooden pallet dimensions [cm³]:** Length, width, and height of the right wooden pallet.

### Mother Pallets Information
- **Mother pallets density [g/cm³]:** Material density of the mother pallets (large supporting pallets).
- **Mother pallets dimensions [cm³]:** Length, width, and height of the mother pallets.

### Dose Information
- **Units (Gy/h, kGy/h, Gy, kGy):** Units for measuring dose rates or total doses.
- **Minimum dose:** The minimum dose threshold required for the experiment.

### Purpose Information
- **Purpose of the test:** A description or identifier for the purpose of each test.

### 0-D Dose Mappings Information
- **List of n 0-D dose mapping points (X1, Y1, Z1), …, (Xn, Yn, Zn):** List of specific points in space where the dose is to be measured (0-D mappings).

### 1-D Dose Mappings Information
- **X-mapping (V or X):** Whether to compute 1D dose mapping along the X direction.
- **Y-mapping (V or X):** Whether to compute 1D dose mapping along the Y direction.
- **Z-mapping (V or X):** Whether to compute 1D dose mapping along the Z direction.
- **List of p 1-D mapping points (X1, Y1, Z1), …, (Xp, Yp, Zp):** List of points around which 1D dose profiles will be plotted.


## Additional Notes

- The simulation is computationally intensive, with a total complexity of O(n * sx * sy * px * py), where *n* is the number of experiments, *(sx, sy)* represents the number of discretized intervals for the source in the X and Y directions, and *(px, py)* represents the number of discretized intervals for the product in the X and Y directions. Those discretized intervals depend on the source/product XY-dimensions and XY-resolutions.
- In case you have (10,10) source resolution and (10,10,10) product resolution : expect for each experiment less than 50 sec per 1D-mapping and around 5min for Dur computation 
- Ensure that the Python environment is properly set up with all necessary dependencies before running the scripts.

## Contact & Support

For any issues, troubleshooting, or further development requests, please reach out to the project maintainer Guillaume Fontaine.

---

