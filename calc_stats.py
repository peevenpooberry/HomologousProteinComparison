#!//usr/bin/python3

import subprocess
from Bio import AlignIO
import sys

muscle_exe = "/home/ubuntu/Protein_Comparison/MUSCLE/muscle-linux-x86.v5.3"
input_file = "/home/ubuntu/Protein_Comparison/MUSCLE/examples/input_file.fasta"
output_file = "/home/ubuntu/Protein_Comparison/MUSCLE/examples/attempt_output_file.fasta"

try:
    subprocess.run([muscle_exe, "-align", input_file, "-output", output_file], check=True)
    alignment = AlignIO.read(output_file, "fasta")
except subprocess.CalledProcessError as e:
    print(f"Error with MUSCLE command: {e.stderr}")
    sys.exit(1)

print(alignment)