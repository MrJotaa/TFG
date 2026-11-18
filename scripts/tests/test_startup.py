"""Portable smoke tests for start.sh, without Docker or third-party modules.

Run from the repository root: python3 -m unittest discover -s scripts/tests -v
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FAKE_DOCKER = r"""#!/bin/sh
printf '%s\n' "$*" >> "$FAKE_DOCKER_LOG"
case "$*" in
  'compose version' | 'info') exit 0 ;;
  'run '* ) test "${FAKE_PROBE_OK:-0}" = 1; exit $? ;;
  *' exec -T backend python -c '* ) test "${FAKE_TORCH_CUDA_OK:-0}" = 1; exit $? ;;
  *' up '* ) exit 0 ;;
esac
exit 2
"""
FAKE_NVIDIA_SMI = "#!/bin/sh\nexit 0\n"


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'backend').mkdir()
        for filename in ('start.sh', 'compose.yaml', 'compose.gpu.yaml'):
            shutil.copyfile(ROOT / filename, self.root / filename)
        self.fake_bin = self.root / 'bin'
        self.fake_bin.mkdir()
        docker = self.fake_bin / 'docker'
        docker.write_text(FAKE_DOCKER)
        docker.chmod(0o755)
        nvidia = self.fake_bin / 'nvidia-smi'
        nvidia.write_text(FAKE_NVIDIA_SMI)
        nvidia.chmod(0o755)
        self.log = self.root / 'docker.log'
        self.env = os.environ.copy()
        self.env.update({
            'PATH': f'{self.fake_bin}:{os.environ.get("PATH", "")}',
            'FAKE_DOCKER_LOG': str(self.log),
            'FAKE_PROBE_OK': '0',
            'FAKE_TORCH_CUDA_OK': '0',
        })

    def start(self, mode='auto'):
        return subprocess.run(
            ['sh', 'start.sh', mode], cwd=self.root, env=self.env,
            text=True, capture_output=True, check=False,
        )

    def calls(self):
        return self.log.read_text() if self.log.exists() else ''

    def test_cpu_mode_creates_random_secret_and_does_not_probe(self):
        result = self.start('cpu')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex((self.root / 'backend/.env').read_text(), r'JWT_SECRET_KEY=[0-9a-f]{64}')
        self.assertIn('compose -f compose.yaml up -d --build', self.calls())
        self.assertNotIn('run --rm --gpus', self.calls())

    def test_existing_env_is_preserved(self):
        env_file = self.root / 'backend/.env'
        env_file.write_text('JWT_SECRET_KEY=some-existing-secret\nOTHER=value\n')
        self.assertEqual(self.start('cpu').returncode, 0)
        self.assertEqual(env_file.read_text(), 'JWT_SECRET_KEY=some-existing-secret\nOTHER=value\n')

    def test_example_template_secret_is_replaced(self):
        (self.root / 'backend/.env.example').write_text('OTHER=hi\nJWT_SECRET_KEY=replace-me\n')
        self.assertEqual(self.start('cpu').returncode, 0)
        created = (self.root / 'backend/.env').read_text()
        self.assertIn('OTHER=hi', created)
        self.assertRegex(created, r'JWT_SECRET_KEY=[0-9a-f]{64}')
        self.assertNotIn('replace-me', created)

    def test_invalid_existing_env_never_overwritten(self):
        env_file = self.root / 'backend/.env'
        env_file.write_text('OTHER=yes\nJWT_SECRET_KEY=\n')
        self.assertNotEqual(self.start('cpu').returncode, 0)
        self.assertEqual(env_file.read_text(), 'OTHER=yes\nJWT_SECRET_KEY=\n')
        self.assertNotIn(' up ', self.calls())

    def test_auto_falls_back_on_docker_gpu_probe_error(self):
        result = self.start('auto')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('run --rm --gpus all', self.calls())
        self.assertIn('compose -f compose.yaml up -d --build', self.calls())

    def test_gpu_requires_working_runtime(self):
        result = self.start('gpu')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(' up ', self.calls())

    def test_gpu_mode_confirmed(self):
        self.env.update(FAKE_PROBE_OK='1', FAKE_TORCH_CUDA_OK='1')
        result = self.start('gpu')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('compose -f compose.yaml -f compose.gpu.yaml up -d --build', self.calls())
        self.assertIn('exec -T backend python -c', self.calls())

    def test_auto_falls_back_if_pytorch_does_not_see_cuda(self):
        self.env.update(FAKE_PROBE_OK='1', FAKE_TORCH_CUDA_OK='0')
        result = self.start('auto')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('compose -f compose.yaml up -d --force-recreate backend frontend', self.calls())

    def test_gpu_does_not_silently_fall_back_if_pytorch_fails(self):
        self.env.update(FAKE_PROBE_OK='1', FAKE_TORCH_CUDA_OK='0')
        result = self.start('gpu')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('force-recreate backend frontend', self.calls())


if __name__ == '__main__':
    unittest.main()
