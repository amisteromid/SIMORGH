import numpy as np
import torch as pt
from tqdm import tqdm
import h5py
import argparse
from glob import glob

from feature_extraction_dynamic_augmentation_v2 import ProteinGraphDataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Raw XYZ and label input')
    parser.add_argument('-i', '--input', 
                        required=True,
                        help='Input h5 file containing coordinates and labels')
    parser.add_argument('-o', '--output',
                        required=True,
                        help='Output HDF5 file path')
    parser.add_argument('--num-workers',
                        type=int,
                        default=16,
                        help='Number of worker processes for data loading')
    parser.add_argument('--min-sequence-length',
                        type=int,
                        default=16,
                        help='Minimum sequence length to process')
    parser.add_argument('--prefetch-factor',
                        type=int,
                        default=2,
                        help='Number of prefetch factor')
    return parser.parse_args()
    
    
    
    
def process_structures(input_path: str, output_path: str, 
                      num_workers: int, prefetch_factor: int,
                      min_sequence_length: int) -> None:
    dataset = ProteinGraphDataset(h5_path=input_path)
    prefetch = prefetch_factor if num_workers > 0 else None
    dataloader = pt.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=True,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=prefetch
    )
    
    device = pt.device("cuda")
    metadata = []
    with h5py.File(output_path, 'w', libver='latest') as db:
        pbar = tqdm(dataloader, desc="Processing structures")
        
        for neighbor_indices, features, labels, pdb_id in pbar:
            if features==None or neighbor_indices==None or labels == None: continue
            size = len(features[0][0])
            if size < min_sequence_length: continue
            metadata.append({
                    'ID': pdb_id,
                    'size': size,
                })
                
            feature_group = db.create_group(f"data/features/{pdb_id}")
            feature_group.create_dataset("node/seq", data=features[0][0])
            feature_group.create_dataset("node/SCOV_ref", data=features[0][1])
            edge_names = ["R_ref", "SCOD_ref", "D_ref"]
            for name, data in zip(edge_names, features[1]):
                feature_group.create_dataset(f"edge/{name}", data=data)
            feature_group.create_dataset("nn_idx", data=neighbor_indices)
                
            label_group = db.create_group(f"data/labels/{pdb_id}")
            label_group.create_dataset("labels", data=labels)
        
        db['metadata/ID'] = np.array([m['ID'] for m in metadata]).astype(np.bytes_)
        db['metadata/size'] = np.array([m['size'] for m in metadata])
            
            
        
def main():
    args = parse_arguments()
    try:
        process_structures(
            input_path=args.input,
            output_path=args.output,
            num_workers=args.num_workers,
            min_sequence_length=args.min_sequence_length,
            prefetch_factor=args.prefetch_factor
        )
        print(f"Successfully processed structures. Output saved to {args.output}")
    except Exception as e:
        print(f"Error processing structures: {str(e)}")
        raise

if __name__ == "__main__":
    main()
