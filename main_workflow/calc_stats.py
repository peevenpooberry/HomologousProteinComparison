#!/usr/bin/env python3

import subprocess
import sys
import socket
import json
from pathlib import Path
import shutil
from typing import Optional
import argparse
import logging

import numpy as np
import pandas as pd

from Bio import AlignIO
from Bio.PDB import MMCIFParser, PDBParser
from pydantic import BaseModel, Field
from Bio.Align import substitution_matrices

# -------------------------
# Constants
# -------------------------

THREADS = 2

INPUT_DIR = "./Input"
WORK_DIR = "./Work"
OUTPUT_DIR = "./Output"

P2RANK_PATH = "/home/ubuntu/Protein_Comparison/p2rank/p2rank_2.5.1"
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
# Logging / Argparse
# -------------------------

parser = argparse.ArgumentParser(
    description='Summarize FASTQ file and output JSON summary'
)

parser.add_argument(
    '-n', '--sessionname',
    required=True,
    type=str,
    help='The name of the current program session'
)

parser.add_argument(
    '-m', '--musclepath',
    type=str,
    required=True,
    help="The path to the MUSCLE binary or .exe file"
)

parser.add_argument(
    '-p', '--p2rankpath',
    required=True,
    type=str,
    help='The path to the built P2Rank directory'
)

parser.add_argument(
    '-a', '--seqweight',
    type=float,
    default=WEIGHT_SEQUENCE
)

parser.add_argument(
    '-b', '--plddtweight',
    type=float,
    default=WEIGHT_PLDDT
)

parser.add_argument(
    '-c', '--p2rankweight',
    type=float,
    default=WEIGHT_P2RANK
)

parser.add_argument(
    '-l', '--loglevel',
    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
    default='WARNING'
)

parser.add_argument(
    '-t', '--threads',
    type=int,
    default=THREADS
)

parser.add_argument(
    '-o', '--outputdir',
    type=str,
    default=OUTPUT_DIR
)

parser.add_argument(
    '-i', '--inputdir',
    type=str,
    default=INPUT_DIR
)

parser.add_argument(
    '-w', '--workdir',
    type=str,
    default=WORK_DIR
)

parser.add_argument(
    '-u', '--plddtmean',
    type=float,
    default=PLDDT_MEAN
)

parser.add_argument(
    '-s', '--plddtsd',
    type=float,
    default=PLDDT_SD
)

args = parser.parse_args()

format_string = (
    f'[%(asctime)s {socket.gethostname()}] '
    '%(module)s.%(funcName)s:%(lineno)s - %(levelname)s - %(message)s'
)
logging.basicConfig(level=args.loglevel, format=format_string)

# -------------------------
# Classes
# -------------------------

class ProteinFile(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    file_name: str
    file_path: Path

    sequence: list[str]
    sequence_to_alignment_map: dict[int, int] = Field(default_factory=dict)

    PLDDT_per_res: list[float] = Field(default_factory=list)
    PLDDT_scores: list[float] = Field(default_factory=list)

    P2Rank_per_res: list[float] = Field(default_factory=list)
                                        
    final_score_per_res: list[float] = Field(default_factory=list)


class Session(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str

    final_score: list[float] = Field(default_factory=list)

    MSA_path: Optional[Path] = None
    P2Rank_output_path: Optional[Path] = None

    sequence_conservation: list[float] = Field(default_factory=list)
    PLDDT_score_conservation: list[float] = Field(default_factory=list)
    P2Rank_score_conservation: list[float] = Field(default_factory=list)

    proteins: list[ProteinFile] = Field(default_factory=list)

# -------------------------
# Functions
# -------------------------

def compute_features(
        input_dir: str, 
        tmp_work_dir: str, 
        plddt_mean: float, 
        plddt_sd: float,
        muscle_path: str,
        p2rank_path: str
        ) -> dict:
    
    session = Session(name="train")

    proteins = parse_structure_files(input_dir, AMINO_ACID_MAP)
    session.proteins = proteins

    generate_fasta(tmp_work_dir, session)
    alignment = muscle_command(tmp_work_dir, muscle_path, session)

    generate_sequence_alignment_maps(alignment, session)
    henikoff_weights = calculate_henikoff_weights(alignment, session)

    session.sequence_conservation = calculate_seq_conservation(alignment)

    calculate_plddt_scores(session, plddt_mean, plddt_sd)
    session.PLDDT_score_conservation = calculate_plddt_conservation(
        henikoff_weights, alignment, session
    )

    make_ds_file(tmp_work_dir, session)
    p2rank_command(tmp_work_dir, session, p2rank_path, THREADS)
    parse_p2rank_output(tmp_work_dir, session)

    session.P2Rank_score_conservation = calculate_p2rank_conservation(
        henikoff_weights, alignment, session
    )

    return {
        "seq": session.sequence_conservation,
        "plddt": session.PLDDT_score_conservation,
        "p2rank": session.P2Rank_score_conservation,
        "proteins": session.proteins
    }


def parse_structure_files(input_dir: str, aa_map: dict) -> list[ProteinFile]:
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


def generate_fasta(work_dir: str, session: Session) -> AlignIO:
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


def muscle_command(work_dir: str, muscle_path: str, session: Session) -> AlignIO:
    work_dir = Path(work_dir)
    fasta_path = work_dir.joinpath("muscle_input.fasta")
    output_fasta = work_dir.joinpath(f"{session.name}_msa.fasta")

    muscle_log = work_dir.joinpath(f"{session.name}_MUSCLE_log.txt")

    try:
        result = subprocess.run([muscle.path, 
                        "-align", fasta_path, 
                        "-output", output_fasta], 
                       check=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT,
                       text=True)
        with open(muscle_log, "w") as out:
            out.writelines(result.stdout)

        alignment = AlignIO.read(output_fasta, "fasta")

    except subprocess.CalledProcessError as e:
        logging.warning(f"Error with MUSCLE command: {e.stderr}")
        sys.exit(1)

    return alignment


def create_alignment_order_map(alignment: AlignIO) -> dict:
    alignment_name_to_row = {rec.id: i for i, rec in enumerate(alignment)}
    return alignment_name_to_row


def calculate_henikoff_weights(alignment, session):
    protein_to_alignment_row = create_alignment_order_map(alignment)

    n_seqs = len(alignment)
    alignment_length = alignment.get_alignment_length()

    raw_weights = np.zeros(n_seqs, dtype=float)

    for col_idx in range(alignment_length):

        column = list(alignment[:, col_idx])
        non_gap = [r for r in column if r != "-"]

        if not non_gap:
            continue

        counts = Counter(non_gap)
        n_distinct = len(counts)

        for seq_index, residue in enumerate(column):
            if residue == "-":
                continue

            raw_weights[seq_index] += 1.0 / (
                n_distinct * counts[residue]
            )

    total = raw_weights.sum()

    if total == 0:
        raw_weights[:] = 1.0 / n_seqs
    else:
        raw_weights /= total

    henikoff_weights = [
        raw_weights[
            protein_to_alignment_row[protein.file_name]
        ]
        for protein in session.proteins
    ]

    return henikoff_weights


def generate_sequence_alignment_maps(alignment: AlignIO, session: Session):
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


def calculate_seq_conservation(alignment: AlignIO) -> list:
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


def normalize(scores: list) -> list:
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return []

    x_min = arr.min()
    x_max = arr.max()

    if x_max == x_min:
        return np.zeros_like(arr).tolist()

    return ((arr - x_min) / (x_max - x_min)).tolist()


def calculate_plddt_scores(session: Session, mean: float, sd: float):
    denom = 2.0 * (sd ** 2)

    for protein in session.proteins:
        arr = np.asarray(
            protein.PLDDT_per_res,
            dtype=float
        )

        scores = np.exp(
            -((arr - mean) ** 2) / denom
        )

        protein.PLDDT_scores = normalize(scores)


def calculate_plddt_conservation(
    henikoff_weights: list,
    alignment: AlignIO,
    session: Session
) -> list:

    aln_len = alignment.get_alignment_length()

    totals = np.zeros(aln_len, dtype=float)
    weight_sums = np.zeros(aln_len, dtype=float)

    for row, protein in enumerate(session.proteins):

        scores = protein.PLDDT_scores
        seq_to_align = protein.sequence_to_alignment_map
        weight = henikoff_weights[row]

        for seq_idx, score in enumerate(scores):

            aln_idx = seq_to_align[seq_idx]

            totals[aln_idx] += score * weight
            weight_sums[aln_idx] += weight

    result = np.divide(
        totals,
        weight_sums,
        out=np.zeros_like(totals),
        where=weight_sums > 0
    )

    return result.tolist()



def make_ds_file(work_dir: str, session: Session):
    work_dir = Path(work_dir)
    ds_path = work_dir.joinpath(f"{session.name}_p2rank.ds")

    protein_paths = [protein.file_path for protein in session.proteins]

    with open(ds_path, "w") as file:
        for path in protein_paths:
            file.write(f"{path}\n")


def p2rank_command(work_dir: str, session: Session, p2rank_path: str, threads: int):
    work_dir = Path(work_dir)
    ds_path = work_dir / f"{session.name}_p2rank.ds"
    output_path = work_dir.joinpath("P2Rank_Output")

    session.P2Rank_output_path = output_path
    p2rank_log = work_dir.joinpath(f"{session.name}_P2Rank_log.txt")

    try:
        output_path.mkdir(exist_ok=True)
        result = subprocess.run(
        ["./prank", "predict",
        "-threads", str(threads),
        "-o", str(output_path),
        str(ds_path)],
        cwd=Path(p2rank_path),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
        )
        with open(p2rank_log, "w") as out:
            out.writelines(result.stdout)

    except subprocess.CalledProcessError as e:
        logging.warning(f"Error running P2Rank command: {e.stderr}")
        sys.exit(1)


def parse_p2rank_output(work_dir: str, session: Session):
    work_dir = Path(work_dir)
    output_path = work_dir.joinpath("P2Rank_Output")

    for protein in session.proteins:
        path = f"{protein.file_name}.cif_residues.csv"
        full_path = output_path.joinpath(path)

        with open(full_path, "r") as f:
            p2rank_output = pd.read_csv(f, skipinitialspace=True)
        
        res_scores = p2rank_output.loc[:, "probability"]
        protein.P2Rank_per_res = list(res_scores)

def calculate_p2rank_conservation(henikoff_weights: list, alignment: AlignIO, session: Session) -> list:
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


def calculate_final_score(
    session: Session,
    sequence_weight: float,
    plddt_weight: float,
    p2rank_weight: float
    ) -> list:

    seq = np.asarray(normalize(session.sequence_conservation), dtype=float)
    plddt = np.asarray(normalize(session.PLDDT_score_conservation), dtype=float)
    p2rank = np.asarray(normalize(session.P2Rank_score_conservation), dtype=float)

    eps = 1e-10
    seq    = np.clip(seq,    eps, None)
    plddt  = np.clip(plddt,  eps, None)
    p2rank = np.clip(p2rank, eps, None)

    log_score = (
        sequence_weight  * np.log(seq) +
        plddt_weight     * np.log(plddt) +
        p2rank_weight    * np.log(p2rank)
    )

    final_score = np.exp(log_score)

    return normalize(final_score)


def reverse_dict(given_dict: dict) -> dict:
    return {v: k for k, v in given_dict.items()}


def map_final_score_to_proteins(session: Session):

    final_score = session.final_score

    for protein in session.proteins:

        mapped = np.zeros(
            len(protein.sequence),
            dtype=float
        )

        reverse_map = reverse_dict(
            protein.sequence_to_alignment_map
        )

        for aln_idx, score in enumerate(final_score):

            if aln_idx in reverse_map:
                seq_idx = reverse_map[aln_idx]
                mapped[seq_idx] = score

        protein.final_score_per_res = mapped.tolist()


def prepare_output(output_dir: str, work_dir: str, session: Session):

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
    session = Session(name=args.sessionname)

    input_dir = args.args.inputdir
    work_dir = args.workdir
    output_dir = args.outputdir
    
    muscle_path = args.musclepath
    p2rank_path = args.p2rankpath
    threads = args.threads

    logging.info(f"Beginning process: {session.name}")

    # 1. Load all protein structure files
    logging.info(f"Parsing structure files")
    proteins = parse_structure_files(input_dir, AMINO_ACID_MAP)
    session.proteins = proteins
    logging.debug(f"Have parsed {len(session.proteins)} files successfully")

    # 2. Generate FASTA of all proteins for MSA generation
    logging.debug("Generating .fasta for MSA generation")
    generate_fasta(work_dir, session)

    # 3. Generate MSA
    logging.info(f"Generating MSA")
    alignment = muscle_command(work_dir, muscle_path, session)
    session.MSA_path = Path(work_dir).joinpath("muscle_input.fasta")
    logging.debug("Generating sequence to alignment map")
    generate_sequence_alignment_maps(alignment, session)

    # 4. Calculate Henikoff weighting
    logging.info("Calculating Henikoff Weights")
    henikoff_weights = calculate_henikoff_weights(alignment, session)

    # 5. Use BLOSSUM62 matrix for seq conservation score
    logging.info("Calculating Sequence Conservation Score")
    seq_conservation = calculate_seq_conservation(alignment)
    session.sequence_conservation = seq_conservation

    # 6. Calculate PLDDT scores via Guassian Weight
    plddt_mean = args.plddtmean
    plddt_sd = args.plddtsd
    logging.debug(f"Calculating PLDDT Scores using gaussian weight (mean: {plddt_mean}, sd: {plddt_sd})")
    calculate_plddt_scores(session, plddt_mean, plddt_sd)

    # 7. PLDDT score merged via MSA and Henikoff weights
    logging.info("Calculating PLDDT Conservation Score")        
    plddt_conservation = calculate_plddt_conservation(henikoff_weights, alignment, session)
    session.PLDDT_score_conservation = plddt_conservation
    
    # 8. Make .ds file containing paths to all the .pdb/.cif files
    logging.debug("Generating .ds file for P2Rank Command")
    make_ds_file(work_dir, session)

    # 9. P2Rank prediction
    logging.info("Beginning P2Rank binding site prediction")
    p2rank_command(work_dir, session, p2rank_path, threads)

    # 10. Parse P2Rank output
    logging.debug("Parsing P2Rank output")
    parse_p2rank_output(work_dir, session)

    # 11. Calculate Conserved P2Rank score via MSA and Henikoff weights
    logging.info("Calculating P2Rank Conservation Score")
    p2rank_conservation = calculate_p2rank_conservation(henikoff_weights, alignment, session)
    session.P2Rank_score_conservation = p2rank_conservation

    # 12. Final Score Calculation (Final Score: logS = a(logSeq)+b(logPLDDT)+c(logP2Rank))
    weight_seq = args.seqweight
    weight_plddt = args.plddtweight
    weight_p2rank = args.p2rankweight
    logging.info(f"Calculating Final Score (a: {weight_seq}, b: {weight_plddt}, c: {weight_p2rank})")
    final_score = calculate_final_score(session, weight_seq, weight_plddt, weight_p2rank)
    session.final_score = final_score

    # 13. Mapping to each residue in original proteins
    logging.debug("Mapping final score to protein residues")
    map_final_score_to_proteins(session)

    # 14. Prepare output files
    logging.debug("Preparing Output and clearing Working Directory")
    prepare_output(output_dir, work_dir, session)

    logging.info(f"Workflow Complete!")


if __name__ == "__main__":
    main()