"""
Main Application Entry Point.

This script provides a Command Line Interface (CLI) to orchestrate the entire
Sim-to-Real Military Object Detection pipeline. It allows the user to execute
individual steps (Data Prep, Training, Evaluation) sequentially.
"""

import os
import sys

def main():
    """Displays the interactive menu and executes selected modules."""
    while True:
        print("\n" + "="*60)
        print(" 🚀 Sim2Real Military Detection System - Control Panel")
        print("="*60)
        print(" 1. [Data Prep]  Convert XML to YOLO Format & Generate Config")
        print(" 2. [Train]      Start Sim-to-Real Domain Adaptation")
        print(" 3. [Eval]       Evaluate Model Metrics (mAP)")
        print(" 4. [Visuals]    Generate Training Reports & Plots")
        print(" 5. [Demo]       Run Visual Comparison (Zero-Shot vs Tuned)")
        print(" 0. [Exit]       Close Application")
        print("-" * 60)
        
        choice = input(" Enter your choice (0-5): ").strip()
        
        if choice == '0':
            print("Exiting application. Goodbye!")
            break
            
        elif choice == '1':
            print("\n>>> Step 1: Initializing Data Preparation...")
            os.system("python src/dataset_prep.py")
            print("\n>>> Generating YOLO Configuration...")
            os.system("python src/create_yaml.py")
            
        elif choice == '2':
            print("\n>>> Step 2: Starting Training Pipeline...")
            os.system("python src/train_domain_adapt.py")
            
        elif choice == '3':
            print("\n>>> Step 3: Running Evaluation...")
            os.system("python src/evaluate.py")
            
        elif choice == '4':
            print("\n>>> Step 4: Organizing Visuals...")
            os.system("python src/visualize.py")
            
        elif choice == '5':
            print("\n>>> Step 5: Running Demo Comparison...")
            os.system("python compare_results.py")
            
        else:
            print("❌ Invalid choice. Please enter a number between 0 and 5.")

if __name__ == "__main__":
    main()