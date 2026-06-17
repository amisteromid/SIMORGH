import os
from glob import glob
from pymol import cmd
import sys
from tqdm import tqdm
import numpy as np
import h5py
import multiprocessing
from multiprocessing import Pool
from Bio import pairwise2

def parse_fasta(fasta_file):
    """Parse fasta file and return dict of {id: (sequence, labels)}"""
    data = {}
    with open(fasta_file, 'r') as f:
        current_id = None
        current_seq = None
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id and current_seq:
                    data[current_id] = (current_seq, None)
                current_id = line[1:]
                current_seq = None
            elif current_seq is None:
                current_seq = line
            else:
                # This is the label line
                labels = np.array([int(c) for c in line], dtype=np.int8)
                data[current_id] = (current_seq, labels)
                current_id = None
                current_seq = None
        # Handle last entry if no labels
        if current_id and current_seq:
            data[current_id] = (current_seq, None)
    return data

def normalize_pdb_id(pdb_id):
    """Normalize PDB ID to match FASTA format (handles ds3 duplicate naming)"""
    # Handle ds3 pattern: 1a0f_A_1a0f_A -> 1a0f_A
    if '_' in pdb_id:
        parts = pdb_id.split('_')
        # If pattern is XXXX_Y_XXXX_Y, extract XXXX_Y
        if len(parts) == 4 and parts[0] == parts[2] and parts[1] == parts[3]:
            return f"{parts[0]}_{parts[1]}"
    return pdb_id

def map_label(seq_unlabeled, seq_labeled, original_labels):
    """Map labels from labeled sequence to unlabeled sequence using alignment"""
    # Align sequences (global alignment)
    alignments = pairwise2.align.globalxx(seq_unlabeled, seq_labeled)
    aligned_unlabeled, aligned_labeled, score = alignments[0][0], alignments[0][1], alignments[0][2]
    identity = (score / max(len(seq_unlabeled), len(seq_labeled))) * 100
    if identity < 90:
        print(f"Mapping:\n{aligned_labeled}\n{aligned_unlabeled}\n")
    
    mapped_labels = []
    label_idx = 0
    print (f"Mapping of\n{aligned_labeled}\n{aligned_unlabeled}\n\n")
    for i in range(len(aligned_unlabeled)):
        unlabeled_res = aligned_unlabeled[i]
        labeled_res = aligned_labeled[i]
        
        if unlabeled_res == '-':
            # Deletion in unlabeled seq - skip
            label_idx += 1
            continue
            
        if labeled_res == '-':
            # Insertion in unlabeled seq - assign label 0
            mapped_labels.append(0)
            
        elif unlabeled_res != labeled_res:
            # Mismatch - print and map label
            print(f"Mismatch at position {len(mapped_labels)}: {labeled_res} -> {unlabeled_res}")
            mapped_labels.append(original_labels[label_idx])
            label_idx += 1
            
        else:
            # Match - map label directly
            mapped_labels.append(original_labels[label_idx])
            label_idx += 1
    
    return np.array(mapped_labels, dtype=np.int8)

def calculate_pseudo_cb(ca_coord, n_coord, c_coord):
    """Calculate pseudo CB coordinate for glycine"""
    if n_coord is not None and c_coord is not None:
        v_ca_n = n_coord - ca_coord
        v_ca_c = c_coord - ca_coord
        v_sum = v_ca_n + v_ca_c
        v_cb_dir = -v_sum / np.linalg.norm(v_sum)
        pseudo_cb = ca_coord + v_cb_dir * 1.52
        return pseudo_cb
    return None

def process_single_pdb(pdb_file):
    """Process a single PDB file with multiple models and return coordinates and sequence"""
    pdb_filename = os.path.basename(pdb_file)
    pdb_id = os.path.splitext(pdb_filename)[0]
    
    try:
        cmd.reinitialize()
        cmd.load(pdb_file, "target", multiplex=1)
        
        all_model_coords = []
        ref_sequence = None
        ref_residues = None
        
        model_names = cmd.get_names()
        
        for model_no, model in enumerate(model_names):
            cmd.set_name(model, 'target')
            
            target_sequence = ''.join(
                [i for i in cmd.get_fastastr(f"target and polymer.protein").split('\n') if '>' not in i]
            ).replace('?', 'X')
            
            target_residues = []
            ca_coords = []
            cmd.iterate_state(1, f"target and polymer.protein and name CA", 
                              "target_residues.append((int(resi), chain)); ca_coords.append([x,y,z])",
                              space=locals())
            
            if model_no == 0:
                ref_sequence = target_sequence
                ref_residues = target_residues.copy()
            else:
                if target_sequence != ref_sequence:
                    print(f"Warning: Model {model_no} sequence mismatch in {pdb_id}")
                    cmd.delete("target")
                    continue
                if target_residues != ref_residues:
                    print(f"Warning: Model {model_no} residue order mismatch in {pdb_id}")
                    cmd.delete("target")
                    continue
                    
            cb_coords = []
            all_atoms = {}
            cmd.iterate_state(1, f"target and polymer.protein and (name CA or name CB or name N or name C)", 
                             "all_atoms.setdefault((int(resi), chain), {})[name] = [x,y,z]",
                             space=locals())
            
            for (resi, chain) in target_residues:
                if (resi, chain) in all_atoms:
                    atoms = all_atoms[(resi, chain)]
                    if 'CB' in atoms:
                        cb_coords.append(atoms['CB'])
                    elif 'CA' in atoms and 'N' in atoms and 'C' in atoms:
                        ca_coord = np.array(atoms['CA'])
                        n_coord = np.array(atoms['N'])
                        c_coord = np.array(atoms['C'])
                        pseudo_cb = calculate_pseudo_cb(ca_coord, n_coord, c_coord)
                        if pseudo_cb is not None:
                            cb_coords.append(pseudo_cb.tolist())
                        else:
                            cb_coords.append([0.0, 0.0, 0.0])
                    else:
                        cb_coords.append([0.0, 0.0, 0.0])
                else:
                    cb_coords.append([0.0, 0.0, 0.0])
            
            combined_coords = np.zeros((len(ca_coords), 2, 3), dtype=np.float32)
            combined_coords[:, 0, :] = np.array(ca_coords, dtype=np.float32)
            combined_coords[:, 1, :] = np.array(cb_coords, dtype=np.float32)
            
            all_model_coords.append(combined_coords)
            
            cmd.delete("target")
        
        if len(all_model_coords) == 0 or ref_sequence is None:
            print(f"No valid models found in {pdb_filename}")
            return None, None, None
               
        coordinates_array = np.stack(all_model_coords, axis=0)
        
        return pdb_id, coordinates_array, ref_sequence
        
    except Exception as e:
        print(f"Error processing {pdb_filename}: {e}")
        return None, None, None

def process_pdb_files_with_labels(pdb_directory, fasta_file, output_h5="protein_data.h5", num_processes=None):
    
    # Parse fasta labels
    print("Parsing FASTA labels...")
    label_data = parse_fasta(fasta_file)
    print(f"Loaded {len(label_data)} entries from FASTA")
    
    pdb_files = sorted(glob(os.path.join(pdb_directory, "*.pdb")))
    
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Processing {len(pdb_files)} PDB files using {num_processes} processes...")
    
    all_data = []
    
    with tqdm(total=len(pdb_files), desc="Processing PDB files") as pbar:
        with Pool(processes=num_processes) as pool:
            for pdb_id, coordinates, sequence in pool.imap_unordered(process_single_pdb, pdb_files):
                if pdb_id is not None:
                    all_data.append((pdb_id, coordinates, sequence))
                pbar.update(1)
    
    # Map and validate sequences with labels
    print(f"\nMapping sequences with labels...")
    stats = {'no_match': [], 'seq_mismatch_mapped': [], 'success': 0}
    
    with h5py.File(output_h5, 'w') as f:
        for pdb_id, coordinates, seq_pdb in tqdm(all_data, desc="Writing to HDF5"):
            # Normalize the PDB ID (handles ds3 duplicates)
            normalized_id = normalize_pdb_id(pdb_id)
            
            # Try to find matching label entry with various formats
            label_entry = None
            for candidate in [normalized_id, normalized_id.lower(), normalized_id.replace('_', '-')]:
                if candidate in label_data:
                    label_entry = label_data[candidate]
                    break
            
            if label_entry is None:
                print (f"{candidate} label not found!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                stats['no_match'].append(pdb_id)
                continue
            
            seq_label, original_labels = label_entry
            
            # Handle sequence mismatch with alignment
            if seq_pdb != seq_label:
                print(f"\nSequence mismatch for {normalized_id}, attempting alignment...")
                if original_labels is not None:
                    labels = map_label(seq_pdb, seq_label, original_labels)
                    stats['seq_mismatch_mapped'].append(normalized_id)
                else:
                    stats['no_match'].append(pdb_id)
                    continue
            else:
                labels = original_labels
            
            # Create group and save data (use normalized ID as key)
            if normalized_id in f:
                continue
            grp = f.create_group(normalized_id)
            grp.create_dataset('sequence', data=seq_pdb.encode('utf-8'))
            grp.create_dataset('coordinates', data=coordinates, compression='gzip')
            
            if labels is not None:
                grp.create_dataset('interface_labels', data=labels)
            
            grp.attrs['num_models'] = coordinates.shape[0]
            grp.attrs['num_residues'] = coordinates.shape[1]
            grp.attrs['original_filename'] = pdb_id  # Store original for reference
            
            stats['success'] += 1
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Processing Summary:")
    print(f"  Successfully processed: {stats['success']}")
    print(f"  Mapped with alignment: {len(stats['seq_mismatch_mapped'])}")
    print(f"  No label match: {len(stats['no_match'])}")
    print(f"{'='*60}")
    
    if stats['seq_mismatch_mapped']:
        print(f"\nMapped with alignment examples: {stats['seq_mismatch_mapped'][:5]}")
    if stats['no_match']:
        print(f"No match examples: {stats['no_match'][:5]}")
    
    print(f"\nResults written to {output_h5}")

if __name__ == "__main__":
    pdb_directory = "/srv/storage/capsid@srv-data2.nancy.grid5000.fr/omokhtari/bbflow_data/all_structures"
    fasta_file = "/srv/storage/capsid@srv-data2.nancy.grid5000.fr/omokhtari/bbflow_data/all_labels_final_crypto_added.fa"
    output_h5 = "/srv/storage/capsid@srv-data2.nancy.grid5000.fr/omokhtari/bbflow_data/coords_labels.h5"
    
    process_pdb_files_with_labels(pdb_directory, fasta_file, output_h5=output_h5)
