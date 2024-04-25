import pymol
from pymol import cmd, stored
import numpy as np
from Bio.PDB import PDBParser, Selection
import warnings
from Bio import BiopythonWarning

def load_backbone(file_path):
    """ Load a PDB file and return the coordinates of the atoms in a numpy array. """
    parser = PDBParser()
    structure = parser.get_structure('PDB', file_path)
    atoms = Selection.unfold_entities(structure, 'A')  # Get all atoms
    coords = np.array([atom.get_coord() for atom in atoms if atom.get_name() == 'CA'])  # Consider CA for simplicity
    return coords

def load_structure(file_path):
    """ Load a PDB file and return the coordinates of all atoms in a numpy array. """
    parser = PDBParser()
    structure = parser.get_structure('PDB', file_path)
    atoms = Selection.unfold_entities(structure, 'A')  # Get all atoms
    coords = np.array([atom.get_coord() for atom in atoms])
    return coords

def gdt(model_coords, ref_coords, thresholds=[0.5, 1, 2, 4]):
    """Calculate the Global Distance Test (GDT) score."""
    gdt_scores = []
    for threshold in thresholds:
        matches = np.sqrt(np.sum((model_coords - ref_coords) ** 2, axis=1)) < threshold
        gdt_scores.append(np.mean(matches))
    return np.mean(gdt_scores)

# Initialize PyMOL
pymol.finish_launching(['pymol', '-cq'])  # '-cq' starts PyMOL without the GUI


if __name__ == '__main__':
    warnings.simplefilter('ignore', BiopythonWarning)

    # Paths to PDB files
    pdb1 = 'C:/Users/nicaz/Desktop/Carpeta de Escritorio/Formación Universitaria/Postdoc/Artículos/ProteinDesingWithStructureGPT/InpaintingExperiment/+A0A023I7V4/AlignedStructures/mutant4vsATPsynthase/alignedATPsynthase.pdb'
    pdb2 = 'C:/Users/nicaz/Desktop/Carpeta de Escritorio/Formación Universitaria/Postdoc/Artículos/ProteinDesingWithStructureGPT/InpaintingExperiment/+A0A023I7V4/AlignedStructures/mutant4vsATPsynthase/alignedMutant4.pdb'

    aligned_model_coords = load_backbone(pdb1)
    aligned_ref_coords = load_backbone(pdb2)

    gdt_score = gdt(aligned_model_coords, aligned_ref_coords)
    print("GDT score:", round(gdt_score * 100, 3))

