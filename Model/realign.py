from Bio.PDB import PDBParser, PDBIO, Superimposer
import argparse


def realign_nmr_pdb(input_pdb_file, output_pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('NMR_Structure', input_pdb_file)

    models = list(structure.get_models())
    reference_model = models[0]  # Choose the first model as reference

    # Use CA atoms for superimposition
    ref_atoms = [atom for atom in reference_model.get_atoms() if atom.id == 'CA']

    super_imposer = Superimposer()

    for model in models[1:]:  # Skip the first model since it's the reference
        alt_atoms = [atom for atom in model.get_atoms() if atom.id == 'CA']

        # Ensure the number of atoms matches in both models
        if len(ref_atoms) == len(alt_atoms):
            super_imposer.set_atoms(ref_atoms, alt_atoms)
            super_imposer.apply(model.get_atoms())

    # Writing the aligned models to a new PDB file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_file)


if __name__ == "__main__":
    # Setup argparse for better command-line input handling
    parser = argparse.ArgumentParser(description="align conformations of given multi-model pdb file")
    parser.add_argument("-i", "--input", required=True, help="Input PDB file")
    parser.add_argument("-o", "--output", required=False, help="Output PDB file (optional)")
    args = parser.parse_args()

    input_pdb_file = args.input
    output_pdb_file = args.output

    # Generates a clean output filename (e.g., "protein.pdb" -> "protein_realigned.pdb")
    if not output_pdb_file:
        if input_pdb_file.endswith('.pdb'):
            output_pdb_file = input_pdb_file.replace('.pdb', '_realigned.pdb')
        else:
            output_pdb_file = input_pdb_file + '_realigned.pdb'

    try:
        realign_nmr_pdb(input_pdb_file, output_pdb_file)
        print(f"Successfully processed! Output saved as: {output_pdb_file}")
    except Exception as e:
        print(f"Error processing {input_pdb_file}: {str(e)}")
