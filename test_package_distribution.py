#!/usr/bin/env python3
"""
Test script to verify data files are included in package distribution.
This checks if checkpoints and other data files are accessible after pip install.
"""

import sys
from pathlib import Path


def test_package_data_accessibility():
    """Test if package data files are accessible."""

    print("=" * 80)
    print("Package Data Accessibility Test")
    print("=" * 80)
    print()

    try:
        import clip_cues

        print(f"✓ clip_cues imported successfully")
        print(f"  Version: {clip_cues.__version__}")
        print(f"  Location: {clip_cues.__file__}")
        print()

        # Determine package directory
        package_file = Path(clip_cues.__file__)
        package_dir = package_file.parent
        repo_root = package_dir.parent.parent  # src/clip_cues -> src -> repo_root

        print(f"Package directory: {package_dir}")
        print(f"Repository root (assumed): {repo_root}")
        print()

        # Test various possible data locations
        data_locations = [
            ("Relative to repo root", repo_root / "data"),
            ("Inside package", package_dir / "data"),
            ("Share directory (system)", Path(sys.prefix) / "share" / "clip-cues" / "data"),
            ("Site-packages data", package_dir.parent / "data"),
        ]

        print("Checking possible data locations:")
        print("-" * 80)

        data_found = False
        working_data_path = None

        for location_name, location_path in data_locations:
            exists = location_path.exists()
            status = "✓ EXISTS" if exists else "✗ NOT FOUND"

            print(f"{status} {location_name}")
            print(f"         {location_path}")

            if exists:
                checkpoints = list((location_path / "checkpoints").glob("*.ckpt"))
                vocab = (location_path / "vocabularies" / "antonyms.csv").exists()
                print(f"         Checkpoints: {len(checkpoints)}")
                print(f"         Vocabulary: {'✓' if vocab else '✗'}")
                if checkpoints:
                    data_found = True
                    working_data_path = location_path

            print()

        print("=" * 80)

        if data_found:
            print(f"✓ SUCCESS: Data files are accessible at:")
            print(f"  {working_data_path}")
            print()
            print("Testing checkpoint loading...")

            try:
                # Try to load a model using the found path
                checkpoint_path = working_data_path / "checkpoints" / "clip_orthogonal_synthclic.ckpt"
                model = clip_cues.load_clip_classifier(str(checkpoint_path))
                print(f"✓ Model loaded successfully: {type(model)}")
                print()
                print("✓ OVERALL: Package data is accessible and functional")
                return 0
            except Exception as e:
                print(f"✗ Failed to load model: {e}")
                print()
                print("⚠ WARNING: Data exists but model loading failed")
                return 1

        else:
            print("✗ CRITICAL: No data files found in any expected location!")
            print()
            print("This means:")
            print("  1. The package was installed without data files")
            print("  2. Users cannot use pre-trained models after pip install")
            print("  3. All README examples will fail")
            print()
            print("Required actions:")
            print("  1. Update pyproject.toml to include data files")
            print("  2. OR move data into src/clip_cues/data/")
            print("  3. Rebuild and reinstall the package")
            print("  4. See CRITICAL_ISSUES.md for detailed solutions")
            return 1

    except ImportError as e:
        print(f"✗ Failed to import clip_cues: {e}")
        print("  Make sure the package is installed: pip install -e .")
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = test_package_data_accessibility()
    sys.exit(exit_code)
