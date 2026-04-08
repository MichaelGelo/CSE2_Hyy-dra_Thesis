#!/bin/bash

#SEARCH PATH FOR FASTA FILES
SEARCH_PATH="/home/dlsu-cse/githubfiles/CSE2_Hyy-dra_Thesis/Resources"

echo "=========================================="
echo " 1. SELECT QUERY FILE"
echo "=========================================="
# Finds all .fasta files
mapfile -t options < <(find "$SEARCH_PATH" -name "*.fasta")

if [ ${#options[@]} -eq 0 ]; then
    echo "No .fasta files found in $SEARCH_PATH"
    echo "Please edit the SEARCH_PATH variable in this script."
    exit 1
fi

# Query Menu
PS3="Select Query Number: "
select query_opt in "${options[@]}"; do
    if [ -n "$query_opt" ]; then
        QUERY_FILE=$query_opt
        break
    else
        echo "Invalid selection. Try again."
    fi
done

echo ""
echo "=========================================="
echo " 2. SELECT REFERENCE FILE"
echo "=========================================="
# Reference Menu
PS3="Select Reference Number: "
select ref_opt in "${options[@]}"; do
    if [ -n "$ref_opt" ]; then
        REF_FILE=$ref_opt
        break
    else
        echo "Invalid selection. Try again."
    fi
done

echo ""
echo "=========================================="
echo " 3. SET WORKLOAD RATIO"
echo "=========================================="
# Loop until valid input
while true; do
    read -p "Enter GPU Ratio (0.1 - 0.9): " GPU_VAL
    # Check if input is a valid float
    if [[ $GPU_VAL =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        break
    else
        echo "Invalid input. Please enter a number like 0.7"
    fi
done

# Calculates FPGA Ratio (1.0 - GPU) using awk for floating point math
FPGA_VAL=$(awk -v gpu="$GPU_VAL" 'BEGIN {printf "%.1f", 1.0 - gpu}')

echo ""
echo "------------------------------------------"
echo " CONFIGURATION SUMMARY:"
echo "------------------------------------------"
echo " Query:  $(basename "$QUERY_FILE")"
echo " Ref:    $(basename "$REF_FILE")"
echo " Ratios: GPU=${GPU_VAL}f | FPGA=${FPGA_VAL}f"
echo "------------------------------------------"
read -p "Press [Enter] to Compile and Run..."

cp config.h config.h.bak


sed -i 's|^\(#define QUERY_FILE[[:space:]]*\)"[^"]*"|\1"'"$QUERY_FILE"'"|' config.h
sed -i 's|^\(#define REFERENCE_FILE[[:space:]]*\)"[^"]*"|\1"'"$REF_FILE"'"|' config.h
sed -i 's|^\(#define GPU_SPEED_RATIO[[:space:]]*\)[0-9.]*f|\1'"$GPU_VAL"'f|' config.h
sed -i 's|^\(#define FPGA_SPEED_RATIO[[:space:]]*\)[0-9.]*f|\1'"$FPGA_VAL"'f|' config.h

echo "config.h updated successfully."

echo "Compiling..."
nvcc -O3 -arch=sm_87 -diag-suppress=177 cpu_utils.c finalcuda.cu -o finalcuda

if [ $? -eq 0 ]; then
    echo "Running Benchmark..."
    ./finalcuda
else
    echo "!!! Compilation Failed !!!"
    mv config.h.bak config.h
    exit 1
fi