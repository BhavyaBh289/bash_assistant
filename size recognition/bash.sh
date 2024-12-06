#!/bin/bash

# Check if folder is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <folder> [output_file]"
  exit 1
fi

# Target folder
FOLDER="$1"

# Output file (default to size_distribution.txt if not provided)
OUTPUT_FILE="${2:-size_distributionexcludingmajorf.txt}"

# Check if the given folder exists
if [ ! -d "$FOLDER" ]; then
  echo "Error: $FOLDER is not a valid directory."
  exit 1
fi

# Function to calculate size
calculate_size() {
  local current_folder="$1"

  # Total size of the current folder
  du -sh "$current_folder" >> "$OUTPUT_FILE"

  # Sizes of subfolders and files
  du -sh "$current_folder"/* 2>/dev/null >> "$OUTPUT_FILE"

  # Recursively call the function for subfolders
  for subfolder in "$current_folder"/*; do
    if [ -d "$subfolder" ]; then
      calculate_size "$subfolder"
    fi
  done
}

# Clear or create the output file
> "$OUTPUT_FILE"

# Start calculation
echo "Calculating size for folder: $FOLDER"
calculate_size "$FOLDER"

# Sort the results by size and save back to the file
sort -rh "$OUTPUT_FILE" -o "$OUTPUT_FILE"

echo "Size distribution saved to $OUTPUT_FILE"
