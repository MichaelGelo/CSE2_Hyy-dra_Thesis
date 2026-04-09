#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "   HomoGPU Launcher (homocuda.cu)"
echo "============================================"
echo

# Ask for Query folder
read -p "Enter QUERY folder path: " query_folder
if [ -z "$query_folder" ]; then
    echo "Error: Query folder cannot be empty!"
    exit 1
fi
if [ ! -d "$query_folder" ]; then
    echo "Error: Query folder does not exist!"
    exit 1
fi

# Ask for Reference folder
read -p "Enter REFERENCE folder path: " ref_folder
if [ -z "$ref_folder" ]; then
    echo "Error: Reference folder cannot be empty!"
    exit 1
fi
if [ ! -d "$ref_folder" ]; then
    echo "Error: Reference folder does not exist!"
    exit 1
fi

echo
echo "Compiling homocuda.cu..."
nvcc -O3 -diag-suppress=177 -arch=sm_75 homocuda.cu -o homocuda

echo "Compilation successful!"
echo
echo "Running homocuda with:"
echo "  Query folder: $query_folder"
echo "  Reference folder: $ref_folder"
echo

# Run the program
./homocuda "$query_folder" "$ref_folder"

echo
echo "Process completed!"
