#!/bin/bash
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
AF3_INPUT="/stor/home/sje779/Work/FinalProject/model_training/af3_input_json"
BOLTZ2_INPUT="/stor/home/sje779/Work/FinalProject/model_training/boltz2_input_yaml"
CHAI1_INPUT="/stor/home/sje779/Work/FinalProject/model_training/chai1_input_fasta"

AF3_OUTPUT="/stor/home/sje779/Work/FinalProject/model_training/Generated_Models/af3"
BOLTZ2_OUTPUT="/stor/home/sje779/Work/FinalProject/model_training/Generated_Models/boltz2"
CHAI1_OUTPUT="/stor/home/sje779/Work/FinalProject/model_training/Generated_Models/chai1"

AF3_WEIGHTS="/stor/home/sje779/Work/Clottein_folding/af3/af3_weights"

# ── Functions ─────────────────────────────────────────────────────────────────

af3() {
    local input_dir="$1"
    local output_dir="$2"
    local gpu="$3"

    mkdir -p "$output_dir"

    for input_path in "$input_dir"/*.json; do
        local filename name
        filename=$(basename "$input_path")
        name="${filename%.json}"

        local protein_out="$output_dir/$name"
        mkdir -p "$protein_out"

        echo "[AF3] Running $name on GPU $gpu"

        docker run --rm \
            --volume "$input_dir":/root/af_input \
            --volume "$protein_out":/root/af_output \
            --volume "$AF3_WEIGHTS":/root/models \
            --volume "/stor/scratch/Marcotte/dbarth/alphafold3_db":/root/public_databases \
            --gpus "device=$gpu" \
            alphafold3 \
            python run_alphafold.py \
                --json_path=/root/af_input/"$filename" \
                --model_dir=/root/models \
                --output_dir=/root/af_output \
                --jackhmmer_n_cpu=32

        docker run --rm \
            -v "$protein_out":/data \
            alphafold3 \
            chown -R "$(id -u):$(id -g)" /data
    done
}

boltz2() {
    local input_dir="$1"
    local output_dir="$2"
    local gpu="$3"

    mkdir -p "$output_dir"

    for input_path in "$input_dir"/*.yaml; do
        local name
        name=$(basename "$input_path" .yaml)

        local protein_out="$output_dir/$name"
        mkdir -p "$protein_out"

        echo "[BOLTZ2] Running $name on GPU $gpu"

        CUDA_VISIBLE_DEVICES="$gpu" \
        boltz predict "$input_path" \
            --out_dir "$protein_out" \
            --diffusion_samples 5 \
            --use_msa_server \
            --no_kernels
    done
}

chai1_parallel() {
    local input_dir="$1"
    local output_dir="$2"

    local files=("$input_dir"/*.fasta)
    local total=${#files[@]}
    local half=$(( total / 2 ))

    (
        for (( i=0; i<half; i++ )); do
            local name
            name=$(basename "${files[$i]}" .fasta)
            echo "[CHAI1] Running $name on GPU 0"
            mkdir -p "$output_dir/$name"
            CUDA_VISIBLE_DEVICES="0" \
            chai-lab fold \
                --num-trunk-samples 5 \
                --use-msa-server \
                "${files[$i]}" \
                "$output_dir/$name"
        done
    ) &
    local pid0=$!

    (
        for (( i=half; i<total; i++ )); do
            local name
            name=$(basename "${files[$i]}" .fasta)
            echo "[CHAI1] Running $name on GPU 1"
            mkdir -p "$output_dir/$name"
            CUDA_VISIBLE_DEVICES="1" \
            chai-lab fold \
                --num-trunk-samples 5 \
                --use-msa-server \
                "${files[$i]}" \
                "$output_dir/$name"
        done
    ) &
    local pid1=$!

    local status=0
    wait "$pid0" || { echo "[CHAI1] GPU 0 half failed"; status=1; }
    wait "$pid1" || { echo "[CHAI1] GPU 1 half failed"; status=1; }
    return $status
}

# ── Main ──────────────────────────────────────────────────────────────────────

echo "=== Stage 1: AF3 (GPU 0) and Boltz2 (GPU 1) in parallel ==="
af3    "$AF3_INPUT"    "$AF3_OUTPUT"    "0" &
pid_af3=$!
boltz2 "$BOLTZ2_INPUT" "$BOLTZ2_OUTPUT" "1" &
pid_boltz2=$!

# Wait for both to finish before starting chai1
wait "$pid_af3"    || { echo "[ERROR] AF3 failed";    exit 1; }
wait "$pid_boltz2" || { echo "[ERROR] Boltz2 failed"; exit 1; }

echo "=== Stage 2: Chai1 split across GPU 0 and GPU 1 ==="
chai1_parallel "$CHAI1_INPUT" "$CHAI1_OUTPUT"