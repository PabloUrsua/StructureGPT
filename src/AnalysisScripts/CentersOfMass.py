from Bio.PDB.MMCIFParser import MMCIFParser
import numpy as np
import pandas as pd

# Dictionary of standard atomic masses
atomic_masses = {
    'H': 1.008, 'C': 12.01, 'N': 14.01, 'O': 16.00, 'P': 30.97, 'S': 32.07
}

def calculate_residue_center_of_mass(residue):
    mass_total = 0.0
    center_of_mass = np.zeros(3)
    
    for atom in residue:
        if atom.element in atomic_masses:
            mass = atomic_masses[atom.element]
            mass_total += mass
            center_of_mass += mass * atom.get_coord()
    
    if mass_total > 0:
        center_of_mass /= mass_total
    
    return center_of_mass

def calculate_distance(coord1, coord2):
    return np.linalg.norm(coord1 - coord2)

# Load the CIF file
cif_path = "your_protein_structure.cif"  # Replace with your actual CIF file path
parser = MMCIFParser()
structure = parser.get_structure("ID", cif_path)

# Calculate centers of mass for all residues
residues_com = []

model = structure[0]  # Assuming interest in the first model
for chain in model:
    for residue in chain:
        if residue.id[0] == " ":  # Ignore heteroatoms, including water
            com = calculate_residue_center_of_mass(residue)
            residues_com.append((chain.id, residue.id[1], residue.resname, com))

# Calculate distances between all centers of mass
distances_data = []
for i, (chain1, resnum1, resname1, com1) in enumerate(residues_com):
    for chain2, resnum2, resname2, com2 in residues_com[i+1:]:  # Avoid repeating pairs
        distance = calculate_distance(com1, com2)
        distances_data.append({
            "Residue 1 Chain": chain1,
            "Residue 1 Number": resnum1,
            "Residue 1 Name": resname1,
            "Residue 2 Chain": chain2,
            "Residue 2 Number": resnum2,
            "Residue 2 Name": resname2,
            "Distance": distance
        })

# Convert the data into a pandas DataFrame
df_distances = pd.DataFrame(distances_data)

# Specify your Excel file name
excel_file_name = "residue_com_distances.xlsx"

# Write the DataFrame to an Excel file
df_distances.to_excel(excel_file_name, index=False)

print(f"Distances data saved to {excel_file_name}")