#!//usr/bin/python3

import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from Bio import AlignIO
from Bio.PDB import MMCIFParser, PDBParser
from pydantic import BaseModel
from Bio.Align import substitution_matrices

# -------------------------
# Constants
# -------------------------
SESSION_NAME = "SESSION1"

INPUT_DIR = "/home/ubuntu/Protein_Comparison/main_workflow/Input"
WORK_DIR = "/home/ubuntu/Protein_Comparison/main_workflow/Work"
OUTPUT_DIR = "/home/ubuntu/Protein_Comparison/main_workflow/Output"

MUSCLE_EXE = "/home/ubuntu/Protein_Comparison/MUSCLE/muscle-linux-x86.v5.3"
AMINO_ACID_MAP = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

PLDDT_MEAN = 70
PLDDT_SD = 5

# -------------------------
# Classes
# -------------------------

class ProteinFile(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    protein_name: str
    file_path: Path
    sequence: list[str]
    PLDDT_per_res: list[float] = []
    P2Rank_per_res: list[float] = []

class Session(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str
    MSA_path: Optional[str] = None
    sequence_conservation: list[float] = []
    PLDDT_score_conservation: list[float] = []
    P2Rank_score_conservation: list[float] = []
    proteins: list[ProteinFile] = []

# -------------------------
# Functions
# -------------------------

def parse_structure_files(input_dir, aa_map):
    cif_parser = MMCIFParser(QUIET=True)
    pdb_parser = PDBParser(QUIET=True)

    proteins = []

    input_dir = Path(input_dir)
    for file in input_dir.iterdir():
        structure = None
        if file.is_file:
            protein_name = file.stem
            with open(file, "r") as f:
                if str(file).endswith(".pdb"):
                    structure = pdb_parser.get_structure(protein_name, f)
                elif str(file).endswith(".cif"):
                    structure = cif_parser.get_structure(protein_name, f)
        
        if structure:
        # for each structure need to get
            # 1. list of residues as single letters using amino_acid_map
            # 2. list of plddt per residue
            # 3. protein name
            # 4. filepath
            filepath = file
            sequence = []
            plddt_per_res = []
            for model in structure:
                for chain in model:
                    for res in chain:
                        id = res.get_id()
                        hetfield = id[0]
                        if hetfield != " ":
                            continue
                        res_name = res.get_resname()
                        sequence.append(aa_map[res_name])
                        if "CA" in res:
                            ca = res["CA"]
                            plddt = ca.get_bfactor()
                            plddt_per_res.append(plddt)
            
            protein = ProteinFile(protein_name=protein_name,
                                  file_path=filepath,
                                  sequence=sequence,
                                  PLDDT_per_res=plddt_per_res)
            proteins.append(protein)

        else:
            continue
    
    return proteins


def generate_fasta(work_dir, session):
    work_dir = Path(work_dir)
    fasta_path = work_dir.joinpath("muscle_input.fasta")

    proteins = session.proteins

    with open(fasta_path, "w") as file:
        for protein in proteins:
            sequence = ""
            for res in protein.sequence:
                sequence += res

            file.write(f">{protein.protein_name}\n")
            file.write(f"{sequence}\n")


def muscle_command(work_dir, muscle_exe, session):
    work_dir = Path(work_dir)
    fasta_path = work_dir.joinpath("muscle_input.fasta")
    output_fasta = work_dir.joinpath(f"{session.name}_msa.fasta")

    try:
        subprocess.run([muscle_exe, "-align", fasta_path, "-output", output_fasta], check=True)
        alignment = AlignIO.read(output_fasta, "fasta")
    except subprocess.CalledProcessError as e:
        print(f"Error with MUSCLE command: {e.stderr}")
        sys.exit(1)
    
    return alignment


def calculate_henikoff_weights(alignment):
    henikoff_weights = []

    alignment_length = alignment.get_alignment_length()
    for i in range(alignment_length):
        weight = 0
        column = alignment[:, i]
        unique_col = set(column)
        column_lst = list(column)
        for unique_res in unique_col:
            weight += 1 / (len(unique_col) * column_lst.count(unique_res))
        
        henikoff_weights.append(weight)
    return henikoff_weights

# def calculate_seq_conservation(henikoff_weights, blosum, mean, sd):
#     pass

# def make_ds_file(work_dir):
#     pass

# def p2rank_command(work_dir):
#     pass

# def parse_p2rank_output(work_dir):
#     pass

# def calculate_final_score():
#     pass

# -------------------------
# Workflow
# -------------------------

def main():
    session = Session(name= SESSION_NAME)

    # 1. Load all protein structure files
    proteins = parse_structure_files(INPUT_DIR, AMINO_ACID_MAP)
    session.proteins = proteins

    # 2. Generate FASTA of all proteins for MSA generation
    generate_fasta(WORK_DIR, session)

    # 3. Generate MSA
    alignment = muscle_command(WORK_DIR, MUSCLE_EXE, session)
    session.MSA_path = Path(WORK_DIR).joinpath("muscle_input.fasta")

    # 4. Calculate Henikoff weighting (Bio.Align.MultipleSeqAlignment)
    henikoff_weights = calculate_henikoff_weights(alignment)

    # 5. Use BLOSSUM62 matrix for seq conservation score
    

    # # 6. Calculate PLDDT scores via Guassian Weight
    # # 7. PLDDT score merged via MSA and Henikoff weights
    # blosum = substitution_matrices.load("BLOSUM62")
    # seq_conservation = calculate_seq_conservation(henikoff_weights, blosum, PLDDT_MEAN, PLDDT_SD)
    
    
    # # 8. make .ds file containing paths to all the pdb/cif files
    # make_ds_file(WORK_DIR)

    # # 9. P2Rank prediction
    # p2rank_command(WORK_DIR)

    # # 10. Parse P2Rank output
    # parse_p2rank_output(WORK_DIR)

    # # 11. Final Score Calculation (Final Score = [Seq Conservation]^a * [PLDDT Score]^b * [P2Rank]^c)
    # calculate_final_score()

    # # 12. Mapping to each residue in original proteins
    # # 13. Prepare output files

if __name__ == "__main__":
    main()