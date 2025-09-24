#!/usr/bin/env python3
"""
Simple test runner script for the RAG evaluation tool unit tests.

This script provides a convenient way to run the unit tests with different options.
"""

import sys
import subprocess
import argparse


def run_unittest():
    """Run tests using unittest."""
    print("Running tests with unittest...")
    cmd = [sys.executable, "-m", "unittest", "test_eval.py", "-v"]
    return subprocess.run(cmd).returncode


def run_pytest():
    """Run tests using pytest."""
    print("Running tests with pytest...")
    cmd = [sys.executable, "-m", "pytest", "test_eval.py", "-v"]
    return subprocess.run(cmd).returncode


def run_pytest_with_coverage():
    """Run tests using pytest with coverage."""
    print("Running tests with pytest and coverage...")
    cmd = [sys.executable, "-m", "pytest", "test_eval.py", "--cov=eval", "--cov-report=term-missing"]
    return subprocess.run(cmd).returncode


def run_coverage():
    """Run tests with coverage module."""
    print("Running tests with coverage module...")
    # Run tests
    cmd1 = [sys.executable, "-m", "coverage", "run", "-m", "unittest", "test_eval.py"]
    result1 = subprocess.run(cmd1)
    
    if result1.returncode == 0:
        # Generate report
        cmd2 = [sys.executable, "-m", "coverage", "report", "-m"]
        result2 = subprocess.run(cmd2)
        return result2.returncode
    
    return result1.returncode


def main():
    """Main function to handle command line arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run unit tests for eval.py")
    parser.add_argument(
        "--runner", 
        choices=["unittest", "pytest", "pytest-cov", "coverage"],
        default="pytest",
        help="Test runner to use (default: pytest)"
    )
    
    args = parser.parse_args()
    
    # Check if required packages are available
    try:
        if args.runner in ["pytest", "pytest-cov"]:
            import pytest
    except ImportError:
        print("Error: pytest is not installed. Please run: pip install pytest pytest-cov")
        return 1
    
    # Run tests based on selected runner
    if args.runner == "unittest":
        return run_unittest()
    elif args.runner == "pytest":
        return run_pytest()
    elif args.runner == "pytest-cov":
        return run_pytest_with_coverage()
    elif args.runner == "coverage":
        return run_coverage()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())