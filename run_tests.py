#!/usr/bin/env python
"""
Test runner for Ukrainian OCR Pipeline
"""

import os
import sys
import unittest
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def discover_tests(test_pattern='test_*.py', start_dir='tests'):
    """Discover and load tests"""
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=start_dir,
        pattern=test_pattern,
        top_level_dir=str(project_root)
    )
    return suite


def run_unit_tests(verbosity=2):
    """Run unit tests only"""
    print("=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)
    
    suite = discover_tests(start_dir='tests/unit')
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_integration_tests(verbosity=2):
    """Run integration tests only"""
    print("\n" + "=" * 60)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 60)
    
    suite = discover_tests(start_dir='tests/integration')
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_all_tests(verbosity=2):
    """Run all tests"""
    print("=" * 60)
    print("RUNNING ALL TESTS FOR UKRAINIAN OCR PIPELINE")
    print("=" * 60)
    
    suite = discover_tests(start_dir='tests')
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_specific_test(test_name, verbosity=2):
    """Run a specific test module or class"""
    print(f"Running specific test: {test_name}")
    
    suite = unittest.TestLoader().loadTestsFromName(test_name)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def check_dependencies():
    """Check if required dependencies are available"""
    print("Checking dependencies...")
    
    required_packages = [
        'numpy',
        'opencv-python',
        'Pillow',
        'torch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not found")
            missing_packages.append(package)
    
    if missing_packages:
        print("\nMissing packages can be installed with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("All required dependencies found!")
    return True


def setup_test_environment():
    """Set up test environment"""
    print("Setting up test environment...")
    
    # Create test fixtures if needed
    fixtures_dir = project_root / 'tests' / 'fixtures'
    if fixtures_dir.exists():
        try:
            sys.path.append(str(fixtures_dir))
            from create_test_data import setup_test_fixtures
            setup_test_fixtures()
            print("✅ Test fixtures created")
        except Exception as e:
            print(f"⚠️ Warning: Could not create test fixtures: {e}")
    
    print("Test environment setup complete")


def print_test_summary(success, test_type=""):
    """Print test summary"""
    print("\n" + "=" * 60)
    if success:
        print(f"✅ {test_type}TESTS PASSED")
        print("All tests completed successfully!")
    else:
        print(f"❌ {test_type}TESTS FAILED")
        print("Some tests failed. Check the output above for details.")
    print("=" * 60)


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description='Run tests for Ukrainian OCR Pipeline')
    parser.add_argument(
        '--unit',
        action='store_true',
        help='Run unit tests only'
    )
    parser.add_argument(
        '--integration',
        action='store_true',
        help='Run integration tests only'
    )
    parser.add_argument(
        '--test',
        type=str,
        help='Run specific test (module.class.method)'
    )
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check dependencies and exit'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=1,
        help='Increase verbosity (use -v, -vv, or -vvv)'
    )
    parser.add_argument(
        '--no-setup',
        action='store_true',
        help='Skip test environment setup'
    )
    
    args = parser.parse_args()
    
    # Convert verbose count to unittest verbosity
    verbosity = min(args.verbose, 3)
    
    # Check dependencies if requested
    if args.check_deps:
        deps_ok = check_dependencies()
        sys.exit(0 if deps_ok else 1)
    
    # Setup test environment unless skipped
    if not args.no_setup:
        setup_test_environment()
    
    success = False
    
    try:
        if args.test:
            # Run specific test
            success = run_specific_test(args.test, verbosity)
            print_test_summary(success, f"SPECIFIC ({args.test}) ")
            
        elif args.unit:
            # Run unit tests only
            success = run_unit_tests(verbosity)
            print_test_summary(success, "UNIT ")
            
        elif args.integration:
            # Run integration tests only
            success = run_integration_tests(verbosity)
            print_test_summary(success, "INTEGRATION ")
            
        else:
            # Run all tests (default)
            unit_success = run_unit_tests(verbosity)
            integration_success = run_integration_tests(verbosity)
            success = unit_success and integration_success
            
            print_test_summary(success, "ALL ")
            
            if not success:
                if not unit_success:
                    print("❌ Unit tests failed")
                if not integration_success:
                    print("❌ Integration tests failed")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
        success = False
    
    except Exception as e:
        print(f"\n\n❌ Test runner error: {e}")
        success = False
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()