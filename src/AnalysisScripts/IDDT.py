import numpy as np
from Bio.PDB import PDBParser, Selection

def load_structure(file_path):
    """ Load a PDB file and return the coordinates of the atoms in a numpy array. """
    parser = PDBParser()
    structure = parser.get_structure('PDB', file_path)
    atoms = Selection.unfold_entities(structure, 'A')  # Get all atoms
    coords = np.array([atom.get_coord() for atom in atoms if atom.get_name() == 'CA'])  # Consider CA for simplicity
    return coords

def calculate_distances(coords):
    """ Calculate the pairwise distances between a set of coordinates. """
    dist = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=-1))
    return dist

def lddt_score(model_coords, ref_coords, thresholds=[0.5, 1, 2, 4], inclusion_radius=15):
    """ Compute the lDDT score between model and reference coordinates. """
    model_distances = calculate_distances(model_coords)
    ref_distances = calculate_distances(ref_coords)

    # Mask to consider only distances within the inclusion radius and not in the same residue
    mask = (ref_distances <= inclusion_radius) & (
                np.abs(np.arange(len(ref_coords))[:, None] - np.arange(len(ref_coords))[None, :]) > 1)

    scores = []
    for threshold in thresholds:
        preserved = np.abs(model_distances - ref_distances) <= threshold
        score = np.sum(preserved & mask) / np.sum(mask)
        scores.append(score)

    return np.mean(scores)

if __name__ == '__main__':
    # Paths to PDB files
    pdb1 = 'C:/Users/nicaz/Desktop/Carpeta de Escritorio/Formación Universitaria/Postdoc/Artículos/ProteinDesingWithStructureGPT/InpaintingExperiment/+A0A023I7V4/AF-A0A023I7V4-F1-model_v4.pdb'
    pdb2 = 'C:/Users/nicaz/Desktop/Carpeta de Escritorio/Formación Universitaria/Postdoc/Artículos/ProteinDesingWithStructureGPT/InpaintingExperiment/+A0A023I7V4/bestRanks/mutant1.pdb'
    pdb3 = 'C:/Users/nicaz/Desktop/Carpeta de Escritorio/Formación Universitaria/Postdoc/Artículos/ProteinDesingWithStructureGPT/InpaintingExperiment/+A0A023I7V4/bestRanks/mutant4.pdb'

    # Load structures and calculate lDDT
    model_coords1 = load_structure(pdb1)
    model_coords2 = load_structure(pdb2)
    model_coords3 = load_structure(pdb3)
    score = lddt_score(model_coords3, model_coords2)
    print("lDDT score:", round(score * 100, 3))
