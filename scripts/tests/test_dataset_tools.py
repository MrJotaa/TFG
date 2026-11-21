"""Regression tests for the downloadable example dataset tools (no internet)."""

import csv
import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_tool(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "datasets" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CocoSubsetTests(unittest.TestCase):
    def test_deterministic_selection_and_annotation_integrity(self):
        coco = load_tool("coco_subset_tool", "download_coco_subset.py")
        doc = {
            "images": [{"id": i, "file_name": f"{i:012}.jpg"} for i in range(1, 7)],
            "annotations": [
                {"id": i, "image_id": i, "category_id": 1 + i % 3}
                for i in range(1, 7)
            ],
            "categories": [{"id": i, "name": str(i)} for i in range(1, 4)],
        }
        ids = coco.select_diverse_image_ids(doc, count=3, seed=2026)
        self.assertEqual(ids, coco.select_diverse_image_ids(doc, count=3, seed=2026))
        self.assertEqual(len(ids), 3)
        subset = coco.build_subset(doc, ids)
        self.assertEqual({img["id"] for img in subset["images"]}, set(ids))
        self.assertTrue(all(a["image_id"] in ids for a in subset["annotations"]))
        self.assertEqual(len(subset["categories"]), 3)

    def test_coco_rejects_requested_count_greater_than_available(self):
        coco = load_tool("coco_subset_tool_limits", "download_coco_subset.py")
        doc = {
            "images": [{"id": 1}],
            "annotations": [{"id": 10, "image_id": 1, "category_id": 4}],
            "categories": [{"id": 4}],
        }
        with self.assertRaises(ValueError):
            coco.select_diverse_image_ids(doc, count=2, seed=2026)


class ImagenetteSubsetTests(unittest.TestCase):
    def test_balanced_labels_and_repeatable_manifest(self):
        imagenette = load_tool("imagenette_tool", "prepare_imagenette.py")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for synset in imagenette.SYNSET_TO_LABEL:
                category = base / "source" / "val" / synset
                category.mkdir(parents=True)
                for i in range(2):
                    (category / f"img{i}.jpg").write_bytes(b"fake-image")
            first = imagenette.prepare_subset(
                base / "source", base / "out", 1, 42, False
            )
            self.assertEqual(len(first[2]), 10)
            second = imagenette.prepare_subset(
                base / "source", base / "out", 1, 42, False
            )
            self.assertEqual(first[2], second[2])
            with first[1].open(newline="", encoding="utf-8") as handle:
                manifest = list(csv.DictReader(handle))
            self.assertEqual(len(manifest), 10)
            self.assertEqual({r["label"] for r in manifest}, set(imagenette.SYNSET_TO_LABEL.values()))

    def test_tar_rejects_links(self):
        import tarfile
        imagenette = load_tool("imagenette_tool_tar", "prepare_imagenette.py")
        with tempfile.TemporaryDirectory() as tmp:
            entry = tarfile.TarInfo("some-link")
            entry.type = tarfile.SYMTYPE
            entry.linkname = "/etc/passwd"
            self.assertFalse(imagenette._is_safe_member(Path(tmp), entry))


class ShellLauncherTests(unittest.TestCase):
    def run_launcher(self, *args, env_extra=None):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "docker"
            captured = Path(tmp) / "args.txt"
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$FAKE_DOCKER_ARGS"\n')
            fake.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = tmp + os.pathsep + env.get("PATH", "")
            env["FAKE_DOCKER_ARGS"] = str(captured)
            env.pop("TFG_API_EMAIL", None)
            env.pop("TFG_API_TOKEN", None)
            env.update(env_extra or {})
            run = subprocess.run(
                ["sh", str(ROOT / "scripts" / "datasets.sh"), *args],
                env=env, text=True, capture_output=True, check=False
            )
            return run, captured.read_text().splitlines() if captured.exists() else []

    def test_coco_100_runs_in_docker(self):
        result, argv = self.run_launcher("coco", "100")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--count", argv)
        self.assertIn("100", argv)
        self.assertIn("/tfg-data/coco_100", argv)
        self.assertIn("--no-deps", argv)

    def test_imagenette_500_prepare_does_not_import(self):
        result, argv = self.run_launcher("imagenette", "500")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("50", argv)
        self.assertIn("--prepare-only", argv)

    def test_imagenette_import_requires_login_context(self):
        result, argv = self.run_launcher("imagenette", "100", "import")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(argv)

    def test_import_forwards_env_by_name_not_secret(self):
        result, argv = self.run_launcher(
            "imagenette", "100", "import",
            env_extra={"TFG_API_TOKEN": "sensitive-example-not-real"}
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("sensitive-example-not-real", argv)
        self.assertIn("TFG_API_TOKEN", argv)
        self.assertNotIn("--prepare-only", argv)


if __name__ == "__main__":
    unittest.main()
