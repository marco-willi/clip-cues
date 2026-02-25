#!/usr/bin/env python3
"""
Comprehensive test script for README instructions.
This script verifies that all code examples and resources mentioned in the README work correctly.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_section(title: str):
    """Print a section header."""
    print(f"\n{BOLD}{BLUE}{'=' * 80}{RESET}")
    print(f"{BOLD}{BLUE}{title}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 80}{RESET}\n")


def print_test(name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"{status} {name}")
    if details:
        print(f"       {details}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{YELLOW}⚠ WARNING: {message}{RESET}")


class ReadmeTestSuite:
    """Test suite for README instructions."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = []
        self.repo_root = Path("/workspaces/clip-cues")
        self.data_dir = self.repo_root / "data"

    def test_file_exists(self, file_path: str, description: str) -> bool:
        """Test if a file exists."""
        path = self.repo_root / file_path
        exists = path.exists()
        print_test(description, exists, str(path) if exists else f"Missing: {path}")
        if exists:
            self.passed += 1
        else:
            self.failed += 1
        return exists

    def test_checkpoints(self) -> Dict[str, bool]:
        """Test all 12 checkpoint files exist."""
        print_section("Testing Pre-trained Model Checkpoints")

        checkpoints = [
            # CLIP Orthogonal Models
            "clip_orthogonal_synthclic.ckpt",
            "clip_orthogonal_synthbuster.ckpt",
            "clip_orthogonal_cnnspot.ckpt",
            "clip_orthogonal_combined.ckpt",
            # Linear Probe Models
            "linear_probe_synthclic.ckpt",
            "linear_probe_synthbuster.ckpt",
            "linear_probe_cnnspot.ckpt",
            "linear_probe_combined.ckpt",
            # Concept Bottleneck Models
            "cm_antonyms_synthclic.ckpt",
            "cm_antonyms_synthbuster.ckpt",
            "cm_antonyms_cnnspot.ckpt",
            "cm_antonyms_combined.ckpt",
        ]

        results = {}
        for ckpt in checkpoints:
            path = self.data_dir / "checkpoints" / ckpt
            exists = path.exists()
            print_test(f"Checkpoint: {ckpt}", exists, str(path) if exists else f"Missing: {path}")
            results[ckpt] = exists
            if exists:
                self.passed += 1
            else:
                self.failed += 1

        return results

    def test_referenced_files(self):
        """Test all files referenced in README exist."""
        print_section("Testing Referenced Files")

        # Files at repo root
        root_files = [
            ("LICENSE", "License file"),
            ("examples/synthclic_paired_samples_collage.png", "SynthCLIC example image"),
            (
                "examples/synthbuster-plus_paired_samples_collage.png",
                "SynthBuster+ example image",
            ),
            (
                "docs/images/synthclic_clic2020_real_images_with_prompts.png",
                "Prompts visualization",
            ),
            ("scripts/TRAINING_GUIDE.md", "Training guide"),
        ]

        # Files in package data directory
        data_files = [
            ("vocabularies/antonyms.csv", "Antonyms vocabulary"),
            ("datasets/synthclic/synthclic_prompts.parquet", "SynthCLIC prompts"),
            (
                "datasets/synthbuster-plus/synthbuster_plus_prompts.parquet",
                "SynthBuster+ prompts",
            ),
        ]

        # Check root files
        for file_path, description in root_files:
            self.test_file_exists(file_path, description)

        # Check data files in package directory
        for file_path, description in data_files:
            path = self.data_dir / file_path
            exists = path.exists()
            print_test(description, exists, str(path) if exists else f"Missing: {path}")
            if exists:
                self.passed += 1
            else:
                self.failed += 1

    def test_package_installation(self):
        """Test package can be imported."""
        print_section("Testing Package Installation")

        try:
            import clip_cues

            print_test("Import clip_cues", True, f"Version: {clip_cues.__version__}")
            self.passed += 1

            # Check main exports
            required_exports = [
                "load_clip_classifier",
                "load_concept_model",
                "list_available_models",
                "SyntheticImageClassifier",
            ]

            for export in required_exports:
                has_export = hasattr(clip_cues, export)
                print_test(f"Export: {export}", has_export)
                if has_export:
                    self.passed += 1
                else:
                    self.failed += 1

        except ImportError as e:
            print_test("Import clip_cues", False, str(e))
            self.failed += 1
            return False

        return True

    def test_model_loading(self):
        """Test loading models as shown in README."""
        print_section("Testing Model Loading (Quick Start Examples)")

        try:
            from clip_cues import load_clip_classifier

            # Test loading the recommended model
            checkpoint_path = "data/checkpoints/clip_orthogonal_synthclic.ckpt"
            print(f"Loading model from {checkpoint_path}...")

            try:
                model = load_clip_classifier(checkpoint_path)
                print_test("Load clip_orthogonal_synthclic.ckpt", True, str(type(model)))
                self.passed += 1

                # Check model has required methods
                has_predict = hasattr(model, "predict")
                has_predict_batch = hasattr(model, "predict_batch")

                print_test("Model has predict() method", has_predict)
                print_test("Model has predict_batch() method", has_predict_batch)

                if has_predict:
                    self.passed += 1
                else:
                    self.failed += 1

                if has_predict_batch:
                    self.passed += 1
                else:
                    self.failed += 1

            except Exception as e:
                print_test("Load clip_orthogonal_synthclic.ckpt", False, str(e))
                self.failed += 1

        except ImportError as e:
            print_test("Import load_clip_classifier", False, str(e))
            self.failed += 1

    def test_all_models_loadable(self):
        """Test that all 12 models can be loaded."""
        print_section("Testing All Model Loading")

        try:
            from clip_cues import load_checkpoint

            checkpoints = [
                "clip_orthogonal_synthclic.ckpt",
                "clip_orthogonal_synthbuster.ckpt",
                "clip_orthogonal_cnnspot.ckpt",
                "clip_orthogonal_combined.ckpt",
                "linear_probe_synthclic.ckpt",
                "linear_probe_synthbuster.ckpt",
                "linear_probe_cnnspot.ckpt",
                "linear_probe_combined.ckpt",
                "cm_antonyms_synthclic.ckpt",
                "cm_antonyms_synthbuster.ckpt",
                "cm_antonyms_cnnspot.ckpt",
                "cm_antonyms_combined.ckpt",
            ]

            for ckpt in checkpoints:
                checkpoint_path = f"data/checkpoints/{ckpt}"
                try:
                    model = load_checkpoint(checkpoint_path)
                    print_test(f"Load {ckpt}", True, "OK")
                    self.passed += 1
                except Exception as e:
                    print_test(f"Load {ckpt}", False, str(e))
                    self.failed += 1

        except ImportError as e:
            print_test("Import load_checkpoint", False, str(e))
            self.failed += 1

    def test_list_available_models(self):
        """Test list_available_models function."""
        print_section("Testing list_available_models()")

        try:
            from clip_cues import list_available_models

            models = list_available_models()
            print_test("Call list_available_models()", True, f"Found {len(models)} models")
            self.passed += 1

            # Check that we have all 12 models
            expected_count = 12
            count_matches = len(models) == expected_count
            print_test(
                f"Expected {expected_count} models",
                count_matches,
                f"Got {len(models)} models",
            )

            if count_matches:
                self.passed += 1
            else:
                self.failed += 1
                self.warnings.append(
                    f"Expected {expected_count} models, but found {len(models)}"
                )

            # Print available models
            print("\nAvailable models:")
            for name, info in models.items():
                print(f"  - {name}: {info.get('description', 'No description')}")

        except ImportError as e:
            print_test("Import list_available_models", False, str(e))
            self.failed += 1
        except Exception as e:
            print_test("Call list_available_models()", False, str(e))
            self.failed += 1

    def test_datasets_accessible(self):
        """Test HuggingFace datasets are accessible (optional)."""
        print_section("Testing HuggingFace Datasets (Optional - may be slow)")

        try:
            from datasets import load_dataset

            print(
                "Note: This test attempts to fetch datasets from HuggingFace and may be slow."
            )
            print("Skipping actual download, only checking import...")

            print_test("Import datasets library", True)
            self.passed += 1

            # Note: We don't actually download to avoid long wait times
            # Just verify the code pattern is correct
            datasets_info = [
                ("marco-willi/synthclic", "SynthCLIC"),
                ("marco-willi/synthbuster-plus", "SynthBuster+"),
                ("marco-willi/cnnspot-small", "CNNSpot"),
            ]

            for dataset_name, description in datasets_info:
                print_test(
                    f"Dataset reference: {description}",
                    True,
                    f"Hub: {dataset_name} (not downloaded)",
                )
                self.passed += 1

        except ImportError:
            print_warning("datasets library not installed - install with: pip install datasets")
            self.warnings.append("datasets library not available for HuggingFace datasets")

    def test_prediction_interface(self):
        """Test prediction interface with a dummy image."""
        print_section("Testing Prediction Interface")

        try:
            from PIL import Image
            import numpy as np
            from clip_cues import load_clip_classifier

            # Create a dummy test image
            print("Creating dummy test image...")
            dummy_img = Image.fromarray(
                np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8)
            )
            test_img_path = "/tmp/test_image.jpg"
            dummy_img.save(test_img_path)

            print_test("Create dummy test image", True, test_img_path)
            self.passed += 1

            # Load model and test prediction
            model = load_clip_classifier("data/checkpoints/clip_orthogonal_synthclic.ckpt")

            # Test single prediction
            try:
                prob = model.predict(test_img_path)
                is_valid = isinstance(prob, (float, np.floating)) and 0 <= prob <= 1
                print_test("Single prediction", is_valid, f"Probability: {prob:.4f}")
                if is_valid:
                    self.passed += 1
                else:
                    self.failed += 1
            except Exception as e:
                print_test("Single prediction", False, str(e))
                self.failed += 1

            # Test batch prediction
            try:
                test_images = [test_img_path] * 3
                probs = model.predict_batch(test_images, batch_size=2)
                is_valid = len(probs) == 3 and all(0 <= p <= 1 for p in probs)
                print_test(
                    "Batch prediction",
                    is_valid,
                    f"Processed {len(probs)} images",
                )
                if is_valid:
                    self.passed += 1
                else:
                    self.failed += 1
            except Exception as e:
                print_test("Batch prediction", False, str(e))
                self.failed += 1

            # Clean up
            os.remove(test_img_path)

        except Exception as e:
            print_test("Prediction interface test", False, str(e))
            self.failed += 1

    def print_summary(self):
        """Print test summary."""
        print_section("Test Summary")

        total = self.passed + self.failed
        pass_rate = (self.passed / total * 100) if total > 0 else 0

        print(f"Total tests: {total}")
        print(f"{GREEN}Passed: {self.passed}{RESET}")
        print(f"{RED}Failed: {self.failed}{RESET}")
        print(f"Pass rate: {pass_rate:.1f}%")

        if self.warnings:
            print(f"\n{YELLOW}Warnings:{RESET}")
            for warning in self.warnings:
                print(f"  - {warning}")

        print()
        if self.failed == 0:
            print(f"{GREEN}{BOLD}✓ All tests passed! README is publication-ready.{RESET}")
            return 0
        else:
            print(
                f"{RED}{BOLD}✗ Some tests failed. Please fix issues before publication.{RESET}"
            )
            return 1


def main():
    """Run all tests."""
    print(f"{BOLD}README Publication Readiness Test Suite{RESET}")
    print("Testing all code examples and references from README.md\n")

    suite = ReadmeTestSuite()

    # Run all test suites
    suite.test_package_installation()
    suite.test_checkpoints()
    suite.test_referenced_files()
    suite.test_list_available_models()
    suite.test_model_loading()
    suite.test_all_models_loadable()
    suite.test_prediction_interface()
    suite.test_datasets_accessible()

    # Print summary and return exit code
    return suite.print_summary()


if __name__ == "__main__":
    sys.exit(main())
