#!/usr/bin/env python3

import MDAnalysis as mda
#from openbabel import openbabel
import subprocess
import os
import pathlib as Path
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import re


INPUT_PATH = "structures_cif/Pro1668_wo_MBP-ligand_CYN"
WORK_PATH = "Intermediates"
MINIMUM_PLDDT = 40

# obConversion = openbabel.OBConversion()
# obConversion.SetInAndOutFormats("cif", "pdb")
# mol = openbabel.OBMol()

def parse_directory(input_dir: str, minimum_pLDDT: int, work_dir):
    input_path = Path.Path(input_dir)
    for child in sorted(input_path.iterdir()):
        print(child.name)
        for file in sorted(child.iterdir()):
            print(file.name[:-4])
            #pdb_path = convert_cif_to_pdb(file, work_dir)
            filtered_protein = parse_pdb(file, minimum_pLDDT)
            print(filtered_protein)
            break
            
            
def convert_cif_to_pdb(file: Path.Path, work_dir: Path.Path):
    filename = file.name[:-4]
    command = [
        "obabel",
        "-icif", str(file),
        "-opdb",
        "-O", f"{work_dir}/{filename}.pdb"
    ]
    subprocess.run(command)
    return Path.Path(f"{work_dir}/{filename}.pdb")


def parse_pdb(file: Path.Path, minimum_pLDDT: int):
    protein = mda.Universe(str(file))
    backbone = protein.select_atoms("protein and name CA")
    filtered_backbone = backbone[backbone.tempfactors > minimum_pLDDT]


def calc_RSMD(target, reference):
    pass


def main():
    parse_directory(INPUT_PATH, MINIMUM_PLDDT, WORK_PATH)


if __name__ == "__main__":
    main()
