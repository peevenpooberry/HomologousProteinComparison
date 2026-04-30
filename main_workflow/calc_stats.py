#!//usr/bin/python3

import subprocess
import sys
import json
from pathlib import Path
import shutil
from typing import Optional

import numpy as np
import pandas as pd

from Bio import AlignIO
from Bio.PDB import MMCIFParser, PDBParser
from pydantic import BaseModel
from Bio.Align import substitution_matrices

# -------------------------
# Constants
# -------------------------
THREADS = 2

SESSION_NAME = "SESSION1"

INPUT_DIR = "/home/ubuntu/Protein_Comparison/main_workflow/Input"
WORK_DIR = "/home/ubuntu/Protein_Comparison/main_workflow/Work"
OUTPUT_DIR = "/home/ubuntu/Protein_Comparison/main_workflow/Output"

P2RANK_PATH = "/home/ubuntu/Protein_Comparison/p2rank/p2rank_2.6.0-dev.7"
MUSCLE_EXE = "/home/ubuntu/Protein_Comparison/MUSCLE/muscle-linux-x86.v5.3"

AMINO_ACID_MAP = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

PLDDT_MEAN = 70
PLDDT_SD = 5
WEIGHT_SEQUENCE = 0.3
WEIGHT_PLDDT = 0.2
WEIGHT_P2RANK = 0.5

# -------------------------
# Classes
# -------------------------

class ProteinFile(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    file_name: str
    file_path: Path
    sequence: list[str]
    sequence_to_alignment_map: dict = {}
    PLDDT_per_res: list[float] = []
    PLDDT_scores: list[float] = []
    P2Rank_per_res: list[float] = []
    final_score_per_res: list[float] = []

class Session(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str
    final_score: list[float] = []
    MSA_path: Optional[str] = None
    P2Rank_output_path: Optional[str] = None
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
            file_name = file.stem
            with open(file, "r") as f:
                if str(file).endswith(".pdb"):
                    structure = pdb_parser.get_structure(file_name, f)
                elif str(file).endswith(".cif"):
                    structure = cif_parser.get_structure(file_name, f)
        
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
            
            protein = ProteinFile(file_name=file_name,
                                  file_path=filepath,
                                  sequence=sequence,
                                  PLDDT_per_res=plddt_per_res)
            proteins.append(protein)

        else:
            continue
    
    return proteins


def generate_fasta(work_dir, session):
    work_dir = Path(work_dir)
    work_dir.mkdir(exist_ok=True)
    fasta_path = work_dir.joinpath("muscle_input.fasta")

    proteins = session.proteins

    with open(fasta_path, "w") as file:
        for protein in proteins:
            sequence = ""
            for res in protein.sequence:
                sequence += res

            file.write(f">{protein.file_name}\n")
            file.write(f"{sequence}\n")


def muscle_command(work_dir, muscle_exe, session):
    work_dir = Path(work_dir)
    fasta_path = work_dir.joinpath("muscle_input.fasta")
    output_fasta = work_dir.joinpath(f"{session.name}_msa.fasta")

    try:
        subprocess.run([muscle_exe, "-align", fasta_path, "-output", output_fasta], 
                       check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        alignment = AlignIO.read(output_fasta, "fasta")

    except subprocess.CalledProcessError as e:
        print(f"Error with MUSCLE command: {e.stderr}")
        sys.exit(1)

    return alignment


def create_alignment_order_map(alignment) -> dict:
    alignment_name_to_row = {rec.id: i for i, rec in enumerate(alignment)}
    return alignment_name_to_row


def calculate_henikoff_weights(alignment, session):
    protein_to_alignment_row = create_alignment_order_map(alignment)

    n_seqs = len(alignment)
    alignment_length = alignment.get_alignment_length()
    raw_weights = [0.0] * n_seqs

    for i in range(alignment_length):
        column = alignment[:, i]
        column_lst = list(column)
        non_gap_residues = [r for r in column_lst if r != "-"]
        if not non_gap_residues:
            continue

        unique_col = set(non_gap_residues)
        n_distinct = len(unique_col)
        
        for seq_index, res in enumerate(column_lst):
            if res == "-":
                continue
            raw_weights[seq_index] += 1 / (n_distinct * column_lst.count(res))

    total = sum(raw_weights)
    henikoff_weights = [
        raw_weights[protein_to_alignment_row[protein.file_name]] / total
        for protein in session.proteins
    ]

    return henikoff_weights


def generate_sequence_alignment_maps(alignment, session):
    protein_to_alignment_row = create_alignment_order_map(alignment)

    for protein in session.proteins:
        alignment_row = alignment[protein_to_alignment_row[protein.file_name], :]
        seq_to_align_map = {}

        seq_index = 0
        for align_index, res in enumerate(alignment_row):
            if res != "-":
                seq_to_align_map[seq_index] = align_index
                seq_index += 1
        
        protein.sequence_to_alignment_map = seq_to_align_map


def calculate_seq_conservation(alignment):
    blosum = substitution_matrices.load("BLOSUM62") 

    scores = []
    alignment_length = alignment.get_alignment_length()
    for i in range(alignment_length):
        column_score = []
        column = [r for r in alignment[:, i] if r != "-"]

        for res1 in range(len(column)):
            for res2 in range(res1 + 1, len(column)):
                a, b = column[res1], column[res2]
                column_score.append(blosum[a, b])
        scores.append(float(np.mean(column_score)) if column_score else 0)
    return scores


def calculate_plddt_scores(session, mean, sd):
    for protein in session.proteins:
        
        plddt_scores = []
        for plddt in protein.PLDDT_per_res:
            score = np.exp(-np.power(plddt - mean, 2.) / (2 * np.power(sd, 2.)))
            plddt_scores.append(float(score))
        protein.PLDDT_scores = plddt_scores


def calculate_plddt_conservation(henikoff_weights, alignment, session):
    alignment_length = alignment.get_alignment_length()
    plddt_conservation = [0.0] * alignment_length
    weight_sums = [0.0] * alignment_length

    for row, protein in enumerate(session.proteins):
        scores = protein.PLDDT_per_res
        seq_to_align_map = protein.sequence_to_alignment_map
        weight = henikoff_weights[row]

        for index, score in enumerate(scores):
            alignment_index = seq_to_align_map[index]
            plddt_conservation[alignment_index] += score * weight
            weight_sums[alignment_index] += weight

    plddt_conservation = [
        total / weight_sums[i] if weight_sums[i] > 0 else 0.0
        for i, total in enumerate(plddt_conservation)
    ]

    return plddt_conservation


def make_ds_file(work_dir, session):
    work_dir = Path(work_dir)
    ds_path = work_dir.joinpath(f"{session.name}_p2rank.ds")

    protein_paths = [protein.file_path for protein in session.proteins]

    with open(ds_path, "w") as file:
        for path in protein_paths:
            file.write(f"{path}\n")


def p2rank_command(work_dir, session, p2rank_path, threads):
    work_dir = Path(work_dir)
    ds_path = work_dir / f"{session.name}_p2rank.ds"
    output_path = work_dir.joinpath("P2Rank_Output")

    session.P2Rank_output_path = output_path

    try:
        output_path.mkdir(exist_ok=True)
        subprocess.run(
        ["./prank", "predict",
        "-threads", str(threads),
        "-o", str(output_path),
        str(ds_path)],
        cwd=Path(p2rank_path),
        check=True,
        capture_output=True,
        text=True
        )

    except subprocess.CalledProcessError as e:
        print(f"Error running P2Rank command: {e.stderr}")
        sys.exit(1)


def parse_p2rank_output(work_dir, session, aa_map):
    work_dir = Path(work_dir)
    output_path = work_dir.joinpath("P2Rank_Output")

    for protein in session.proteins:
        path = f"{protein.file_name}.cif_residues.csv"
        full_path = output_path.joinpath(path)

        with open(full_path, "r") as f:
            p2rank_output = pd.read_csv(f, skipinitialspace=True)
        
        res_scores = p2rank_output.loc[:, "probability"]
        protein.P2Rank_per_res = list(res_scores)


def normalize(scores):
    x_min = min(scores)
    x_max = max(scores)
    if x_max == x_min:
        return [0] * len(scores)
    return [(x - x_min) / (x_max - x_min) for x in scores]


def calculate_p2rank_conservation(henikoff_weights, alignment, session):
    p2rank_conservation = [0.0] * alignment.get_alignment_length()
    weight_sums = [0.0] * alignment.get_alignment_length()

    for row_index, protein in enumerate(session.proteins):
        p2rank_score = protein.P2Rank_per_res
        norm_p2rank_score = normalize(p2rank_score)
        seq_to_align_map = protein.sequence_to_alignment_map
        weight = henikoff_weights[row_index]

        
        for index, score in enumerate(norm_p2rank_score):
            alignment_index = seq_to_align_map[index]
            p2rank_conservation[alignment_index] += score * weight
            weight_sums[alignment_index] += weight

    p2rank_conservation = [
        total / weight_sums[i] if weight_sums[i] > 0 else 0.0
        for i, total in enumerate(p2rank_conservation)
    ]

    return p2rank_conservation


def calculate_final_score(session, sequence_weight, plddt_weight, p2rank_weight):
    sequence_conservation = normalize(session.sequence_conservation)
    PLDDT_score_conservation = normalize(session.PLDDT_score_conservation)
    P2Rank_score_conservation = normalize(session.P2Rank_score_conservation)

    final_score = [
        (seq ** sequence_weight) * (plddt ** plddt_weight) * (p2rank ** p2rank_weight)
        for seq, plddt, p2rank 
        in zip(sequence_conservation, PLDDT_score_conservation, P2Rank_score_conservation)
    ]
    
    return normalize(final_score)


def reverse_dict(given_dict):
    return_dict = {}
    for key in given_dict.keys():
        val = given_dict[key]
        return_dict[val] = key

    return return_dict


def map_final_score_to_proteins(session):
    final_score = session.final_score

    for protein in session.proteins:
        final_score_per_res = [0.0] * len(protein.sequence)
        map = protein.sequence_to_alignment_map
        reverse_map = reverse_dict(map)

        for index, score in enumerate(final_score):
            if index in reverse_map.keys():
                final_score_per_res[reverse_map[index]] += score

        protein.final_score_per_res = final_score_per_res


def prepare_output(output_dir, work_dir, session):

    output_dir = Path(output_dir)
    work_dir = Path(work_dir)

    msa_path = Path(session.MSA_path)
    new_msa_path = output_dir.joinpath(f"{session.name}_msa.fasta")

    p2rank_output_path = session.P2Rank_output_path
    new_p2rank_output_path = output_dir.joinpath(f"{session.name}_P2Rank_output")

    program_file_path = output_dir.joinpath(f"{session.name}_summary.json")
    
    protein_files = [
        {
            "file_name": protein.file_name,
            "sequence": protein.sequence,
            "PLDDT_per_residue": protein.PLDDT_per_res,
            "PLDDT_score_per_residue": protein.PLDDT_scores,
            "P2Rank_per_residue": protein.P2Rank_per_res,
            "final_score_per_residue": protein.final_score_per_res
        } for protein in session.proteins
    ]

    result = {
        "final_score": session.final_score,
        "sequence_conservation": session.sequence_conservation,
        "PLDDT_score_conservation": session.PLDDT_score_conservation,
        "P2Rank_score_conservation": session.P2Rank_score_conservation,
        "protein_files": protein_files 
    }

    root = {"result": result}

    with open(program_file_path, "w") as out:
        json.dump(root, out, indent=4, sort_keys=True)

    subprocess.run(["mv", msa_path, new_msa_path], check=True)
    subprocess.run(["mv", p2rank_output_path, new_p2rank_output_path])
    
    for item in work_dir.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)



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
    generate_sequence_alignment_maps(alignment, session)

    # 4. Calculate Henikoff weighting (Bio.Align.MultipleSeqAlignment)
    henikoff_weights = calculate_henikoff_weights(alignment, session)

    # 5. Use BLOSSUM62 matrix for seq conservation score
    seq_conservation = calculate_seq_conservation(alignment)
    session.sequence_conservation = seq_conservation

    # 6. Calculate PLDDT scores via Guassian Weight
    calculate_plddt_scores(session, PLDDT_MEAN, PLDDT_SD)

    # 8. PLDDT score merged via MSA and Henikoff weights        
    plddt_conservation = calculate_plddt_conservation(henikoff_weights, alignment, session)
    session.PLDDT_score_conservation = plddt_conservation
    
    # 9. make .ds file containing paths to all the pdb/cif files
    make_ds_file(WORK_DIR, session)

    # 10. P2Rank prediction
    p2rank_command(WORK_DIR, session, P2RANK_PATH, THREADS)

    # 11. Parse P2Rank output
    parse_p2rank_output(WORK_DIR, session, AMINO_ACID_MAP)

    # 12. P2Rank score merged via MSA and Henikoff weights
    p2rank_conservation = calculate_p2rank_conservation(henikoff_weights, alignment, session)
    session.P2Rank_score_conservation = p2rank_conservation

    # 13. Final Score Calculation (Final Score = [Seq Conservation]^a * [PLDDT Score]^b * [P2Rank]^c)
    final_score = calculate_final_score(session, WEIGHT_SEQUENCE, WEIGHT_PLDDT, WEIGHT_P2RANK)
    session.final_score = final_score

    # 14. Mapping to each residue in original proteins
    map_final_score_to_proteins(session)

    # 15. Prepare output files
    prepare_output(OUTPUT_DIR, WORK_DIR, session)

if __name__ == "__main__":
    main()