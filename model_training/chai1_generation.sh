#!/bin/bash
set -euo pipefail

source /stor/home/sje779/Work/Clottein_folding/chai1/chai_venv/bin/activate

CHAI1_INPUT="/stor/home/sje779/Work/FinalProject/model_training/chai1_input_fasta"
CHAI1_OUTPUT="/stor/home/sje779/Work/FinalProject/model_training/Generated_Models/chai1"

mkdir -p "$CHAI1_OUTPUT"

echo "=== Running Chai1 on GPU 1 ==="

for input_path in "$CHAI1_INPUT"/*.fasta; do
    name=$(basename "$input_path" .fasta)
    echo "[CHAI1] Running $name on GPU 1"
    mkdir -p "$CHAI1_OUTPUT/$name"

    CUDA_VISIBLE_DEVICES="1" \
    chai-lab fold \
        --num-trunk-samples 5 \
        --use-msa-server \
        "$input_path" \
        "$CHAI1_OUTPUT/$name"
done

echo "=== Chai1 complete ==="