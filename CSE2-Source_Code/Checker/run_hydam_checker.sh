#!/bin/bash

echo "============================================"
echo "   HyDam Checker - Interactive Launcher"
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
echo "Compiling HyDam_checker..."
gcc "[HyDam_checker].c" -o "[HyDam_checker]"
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo
echo "Running HyDam_checker with:"
echo "  Query folder: $query_folder"
echo "  Reference folder: $ref_folder"
echo

# Run the program
"./[HyDam_checker]" "$query_folder" "$ref_folder"

echo
echo "Process completed!"
