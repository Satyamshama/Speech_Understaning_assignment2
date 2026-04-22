#!/bin/bash
# Setup script for Speech Assignment 2

echo "==================================="
echo "Speech Assignment 2 Setup Script"
echo "==================================="

# Check if conda is installed
if command -v conda &> /dev/null; then
    echo "Creating conda environment from environment.yml..."
    conda env create -f environment.yml -y
    echo "Conda environment created. Activate with: conda activate speech-assignment2"
else
    echo "Conda not found. Installing using pip..."
    pip install -r requirements.txt
fi

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "To run the pipeline:"
echo "  python pipeline.py"
echo ""
echo "Individual parts can be run with:"
echo "  python Scripts/PA2_Part1_STT.py       # Speech-to-Text with LID"
echo "  python Scripts/PA2_Part2_Phonetic.py  # Phonetic Mapping"
echo "  python Scripts/PA2_Part3_TTS.py       # Text-to-Speech with Voice Cloning"
echo "  python Scripts/PA2_Part4_Adversarial.py # Adversarial Robustness & Spoofing Detection"
