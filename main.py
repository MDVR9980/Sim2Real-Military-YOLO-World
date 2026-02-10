"""
Main Application Entry Point.

This script acts as the central command hub for the Sim2Real Military Object Detection project.
It allows sequential execution of the pipeline:
1. Organize Data
2. Configure
3. Train (Sim-to-Real)
4. Evaluate (Real Test)
5. Visualize Reports
"""

import os
import sys
from pathlib import Path

def main():
    while True:
        print("\n" + "="*50)
        print(" 🛡️  Sim2Real YOLO-World Project Manager")
        print("="*50)
        print(" 1. [Setup] Organize Dataset (Prof's Data)")
        print(" 2. [Setup] Generate YAML Configuration")
        print(" 3. [Run]   Start Training (RTX 5080 Mode)")
        print(" 4. [Test]  Evaluate on Real-World Test Set")
        print(" 5. [Rep]   Generate Visual Reports")
        print(" 0. [Exit]  Quit")
        print("-" * 50)
        
        choice = input("Select an option (0-5): ").strip()

        if choice == '0':
            print("Exiting...")
            sys.exit()

        elif choice == '1':
            print("\n>>> Organizing Dataset...")
            os.system("python src/organize_dataset.py")

        elif choice == '2':
            print("\n>>> Creating Data Config...")
            os.system("python src/create_yaml.py")

        elif choice == '3':
            print("\n>>> WARNING: This starts heavy training.")
            confirm = input("Are you ready? (y/n): ")
            if confirm.lower() == 'y':
                os.system("python src/train_domain_adapt.py")

        elif choice == '4':
            print("\n>>> Running Final Evaluation...")
            os.system("python src/evaluate.py")

        elif choice == '5':
            print("\n>>> Generating Charts and Figures...")
            os.system("python src/visualize.py")

        else:
            print("❌ Invalid option.")

        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()