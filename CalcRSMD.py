#!/usr/bin/env python3

import os
import socket
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from Bio.PDB import MMCIFParser, MMCIF2Dict # type: ignore
import rmsd

# -------------------------
# global variables
# -------------------------

OUTPUT_FILE = "output.csv"
MINIMUM_PLDDT = 70

# -------------------------
# Command/Logging
# -------------------------
parser = argparse.ArgumentParser(
  description='Calculates an RSMD matrix for a given directory of mmCIF protein prediction files'
)
parser.add_argument(
    '-l', '--loglevel',
    required=False,
    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
    default='WARNING',
    help='Set the logging level (default: WARNING)'
)
parser.add_argument(
    '-i', '--inputcif',
    required=True,
    type=str,
    help=f"The name of the input directory with protein prediction files ending in .cif"
)
parser.add_argument(
    '-o', '--output',
    type=str,
    default=OUTPUT_FILE,
    help=f'The name of the output file (default: {OUTPUT_FILE})'
)
parser.add_argument(
    '-p', '--plddt',
    default=MINIMUM_PLDDT,
    help=f'The minimum pLDDT score that an alpha carbon can posess (default: {MINIMUM_PLDDT})',    
    type=int
)

args = parser.parse_args()

format_string = (
    f'[%(asctime)s {socket.gethostname()}] '
    '%(module)s.%(funcName)s:%(lineno)s - %(levelname)s - %(message)s'
)
logging.basicConfig(level=args.loglevel, format=format_string)

# -------------------------
# Functions
# -------------------------
# def get_plddt_from_mmcif(filepath: str):
#     mmcif_dict = MMCIF2Dict.MMCIF2Dict(filepath)
    
#     plddt_map = {}
    
#     #AF3 metrics first
#     if '_ma_qa_metric_local.metric_value' in mmcif_dict:
#         chain_ids = mmcif_dict['_ma_qa_metric_local.label_asym_id']
#         seq_ids   = mmcif_dict['_ma_qa_metric_local.label_seq_id']
#         values    = mmcif_dict['_ma_qa_metric_local.metric_value']
        
#         for chain, seq_id, val in zip(chain_ids, seq_ids, values):
#             if seq_id == "." or val == ".":
#                 continue
#             plddt_map[(chain, int(seq_id))] = float(val)
    
#     # Fall back to B-factor if AF3 field not present
#     return plddt_map


def get_filtered_ca_atoms(structure, filepath: str, min_plddt: int) -> dict:
    ca_atoms = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] != " ":
                    continue
                if "CA" in residue:
                    ca = residue["CA"]
                    seq_id = residue.get_id()[1]
                    plddt = ca.get_bfactor() 
                    if plddt >= min_plddt:
                        key = (chain.id, seq_id)
                        ca_atoms[key] = ca
    return ca_atoms


def get_common_ca_coords(atoms1: dict, atoms2: dict)-> tuple:
    common_keys = sorted(set(atoms1.keys()) & set(atoms2.keys()))
    if len(common_keys) < 3:
        return None, None
    
    coords1 = np.array([atoms1[k].get_vector().get_array() for k in common_keys])
    coords2 = np.array([atoms2[k].get_vector().get_array() for k in common_keys])
    return coords1, coords2


def run_rsmd_comparison(mmcif_dir: str, plddt_threshold:int, output_csv:str):
    parser = MMCIFParser(QUIET=True)

    files = [file for file in os.listdir(mmcif_dir) if file.endswith(".cif")]
    
    structures = {}
    ca_atoms = {}

    logging.info(f"Parsing through {len(files)} files")
    for filename in files:
        name = os.path.splitext(filename)[0]
        path = os.path.join(mmcif_dir, filename)
        try:
            logging.debug(f"Getting structure from {filename}")
            struct = parser.get_structure(name, path)

            logging.debug(f"Filtering alpha carbsons from {filename} by pLDDT {plddt_threshold}")
            atoms = get_filtered_ca_atoms(struct, path, plddt_threshold)
            structures[name] = struct
            ca_atoms[name] = atoms
        except Exception as e:
            logging.warning(f"failed to get alpha carbons from {filename}: {e}")
            continue
    
    if len(ca_atoms.keys()) == 0:
        logging.error(f"Failed to parse files in {mmcif_dir} ending process")
        sys.exit(1)
    
    names = sorted(ca_atoms.keys())
    n = len(names)
    rsmd_matrix = np.full((n, n), np.nan)
    overlap_matrix = np.full((n, n), 0, dtype=int)

    logging.info(f"Successfully parsed {len(ca_atoms.keys())} files")
    logging.info("Beginning to calculate RSMD matrix")
    
    count = 0
    for i, j in combinations(range(n), 2):
        logging.debug(f"Getting common coords for {names[i]} and {names[j]}")
        coords1, coords2 = get_common_ca_coords(ca_atoms[names[i]], ca_atoms[names[j]])
        if coords1 is None:
            logging.debug("Less than 3 common coordinates")
            continue

        logging.debug("Calculating RSMD")
        rmsd_val = rmsd.kabsch_rmsd(coords1, coords2, translate=True)
        n_common = len(coords1)

        rsmd_matrix[i, j] = rmsd_val
        rsmd_matrix[j, i] = rmsd_val
        overlap_matrix[i, j] = n_common
        overlap_matrix[j, i] = n_common

        count += 1
    
    np.fill_diagonal(rsmd_matrix, 0.0)

    logging.info(f"Successfully calculated RSMD values for {count} pairs")
    logging.info(f"Exporting RSMD matrix as {output_csv}")


    df_rsmd = pd.DataFrame(rsmd_matrix, index=names, columns=names)
    df_rsmd.to_csv(output_csv)

    df_overlap = pd.DataFrame(overlap_matrix, index=names, columns=names)
    df_overlap.to_csv(output_csv.replace(".csv", "_overlap.csv"))

    return df_rsmd, df_overlap

# -------------------------
# Main
# -------------------------
def main():
    logging.info(f"Beginning workflow for directory: {args.inputcif}")
    run_rsmd_comparison(args.inputcif, args.plddt, args.output)
    logging.info("Successfully completed workflow!")


if __name__ == "__main__":
    main()