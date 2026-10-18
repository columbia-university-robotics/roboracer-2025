"""Local portability and process-lifecycle tests; ROS is deliberately not needed."""
import contextlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('launcher', ROOT / 'run_sim.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@contextlib.contextmanager
def working_directory(path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class LauncherTests(unittest.TestCase):
    def test_paths_with_spaces_and_bundled_adapter(self):
        with tempfile.TemporaryDirectory(prefix='team workspace ') as tmp:
            ws = Path(tmp) / 'sim_ws'
            config = ws / 'src/planning/config/sim.yaml'
            config.parent.mkdir(parents=True)
            config.write_text('test fixture')
            with working_directory(ws), patch.dict(os.environ, {}, clear=True), patch.object(launcher.shutil, 'which', return_value='/opt/ros/humble/bin/ros2'):
                adapter, found, _ = launcher.runtime_paths()
                self.assertEqual(found, config.resolve())
                self.assertEqual(adapter, ROOT / 'sim_pose_adapter.py')
                with patch.dict(os.environ, {'ROBORACER_PLANNING_CONFIG': str(ws / 'missing.yaml')}):
                    with self.assertRaisesRegex(RuntimeError, 'Planning config not found'):
                        launcher.runtime_paths()

    def test_original_sibling_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / 'sim_ws'
            ws.mkdir()
            config = Path(tmp) / 'f1tenth_ws/src/planning/config/sim.yaml'
            config.parent.mkdir(parents=True)
            config.touch()
            with working_directory(ws), patch.dict(os.environ, {}, clear=True), patch.object(launcher.shutil, 'which', return_value='/opt/ros/humble/bin/ros2'):
                self.assertEqual(launcher.runtime_paths()[1], config.resolve())

    def test_shell_environment_and_workspace_override(self):
        # Fake setup files exercise shell orchestration without importing ROS.
        with tempfile.TemporaryDirectory(prefix='shell fixture ') as tmp:
            ws = Path(tmp) / 'custom workspace'
            (ws / '.venv/bin').mkdir(parents=True)
            (ws / 'install').mkdir()
            (ws / '.venv/bin/activate').write_text('export VENV_TEST=loaded\n')
            (ws / 'install/local_setup.bash').write_text('export OVERLAY_TEST=loaded\n')
            setup = Path(tmp) / 'setup.bash'
            # Provide a fake interpreter to observe the loaded shell environment.
            fake = ws / '.venv/bin/python3'
            fake.write_text('#!/bin/bash\nprintf "%s|%s|%s|%s" "$PWD" "$VENV_TEST" "$OVERLAY_TEST" "$2"\n')
            fake.chmod(0o755)
            (ws / '.venv/bin/activate').write_text('export VENV_TEST=loaded\nexport PATH="$PWD/.venv/bin:$PATH"\n')
            setup.write_text(':\n')
            env = dict(os.environ, ROBORACER_SIM_WS=str(ws), ROBORACER_ROS_SETUP=str(setup))
            env.pop('ROBORACER_VENV', None)
            result = subprocess.run(['bash', str(ROOT / 'run_sim.sh'), '--preflight'], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            location, *values = result.stdout.split('|')
            self.assertEqual(Path(location).resolve(), ws.resolve())
            self.assertEqual(values, ['loaded', 'loaded', '--preflight'])
            setup.unlink()
            result = subprocess.run(['bash', str(ROOT / 'run_sim.sh'), '--drive'], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Required environment file not found', result.stderr)

    def test_child_cleanup_and_failure_reporting(self):
        with tempfile.TemporaryDirectory() as tmp:
            children = launcher.Children(Path(tmp))
            try:
                children.start('follower', [sys.executable, '-c', 'import time; time.sleep(60)'])
                children.start('other', [sys.executable, '-c', 'import time; time.sleep(60)'])
                children.check()
                children.stop({'follower'})
                self.assertIsNotNone(children.items[0][1].poll())
                self.assertIsNone(children.items[1][1].poll())
            finally:
                children.stop()
            self.assertTrue(all(p.poll() is not None for _, p in children.items))
            failed = launcher.Children(Path(tmp))
            failed.start('failed', [sys.executable, '-c', 'raise SystemExit(7)'])
            failed.items[0][1].wait(timeout=5)
            with self.assertRaisesRegex(RuntimeError, r'failed exited \(7\)'):
                failed.check()


if __name__ == '__main__':
    unittest.main()
