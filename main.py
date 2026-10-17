#!/usr/bin/env python3
"""
AI Concentration Tracking System - Unified Launcher
Seamlessly launch either the Modern Web Dashboard (HTML/CSS/JS) or the Desktop OpenCV App.
"""
import sys
import os
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_RUNNER = os.path.join(PROJECT_ROOT, "web", "run_web.py")
DESKTOP_SCRIPT = os.path.join(PROJECT_ROOT, "concentration_tracker", "concen_tracker.py")

def run_web():
    print("\nStarting Web Dashboard (HTML / CSS / JS)...")
    subprocess.run([sys.executable, WEB_RUNNER])

def run_desktop():
    print("\nStarting Desktop OpenCV Concentration Tracker...")
    subprocess.run([sys.executable, DESKTOP_SCRIPT])

def interactive_menu():
    print("=" * 60)
    print("    AI CONCENTRATION TRACKING SYSTEM - UNIFIED LAUNCHER")
    print("=" * 60)
    print("  [1] Launch Modern Web Dashboard (HTML5, CSS3, JS)")
    print("  [2] Launch Desktop OpenCV Application")
    print("  [3] Exit")
    print("-" * 60)
    
    try:
        choice = input("Select an option (1, 2, or 3): ").strip()
        if choice == "1":
            run_web()
        elif choice == "2":
            run_desktop()
        elif choice == "3":
            print("Exiting.")
            sys.exit(0)
        else:
            print("Invalid selection. Defaulting to Web Dashboard...")
            run_web()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="AI Concentration Tracking System Unified Runner")
    parser.add_argument("--web", action="store_true", help="Launch the Web Dashboard directly")
    parser.add_argument("--desktop", action="store_true", help="Launch the Desktop OpenCV application directly")
    args = parser.parse_args()

    if args.web:
        run_web()
    elif args.desktop:
        run_desktop()
    else:
        interactive_menu()

if __name__ == "__main__":
    main()
