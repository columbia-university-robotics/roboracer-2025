"""Foreground simulation supervisor. Stop only children started by this run."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime

TARGETS = ('gym_bridge', 'foxglove_bridge', 'sim_pose_adapter.py',
           'occupancy_grid_planner', 'pure_pursuit_follower',
           'check_planning_goal.py', 'verify_pose_adapter.py',
           'teleop_twist_keyboard')
NODES = {'bridge', 'foxglove_bridge', 'sim_pose_adapter',
         'occupancy_grid_planner', 'pure_pursuit_follower',
         'planning_goal_diagnostic', 'verify_pose_adapter'}


def existing_processes():
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            args = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            # Match executable/script argument names, never arbitrary shell text.
            if any(Path(a).name in TARGETS for a in args if a):
                found.append((entry.name, ' '.join(a for a in args if a)))
        except (OSError, ProcessLookupError):
            pass
    return found


class Children:
    def __init__(self, directory):
        self.directory = directory
        self.items = []

    def start(self, name, args):
        path = self.directory / (name + '.log')
        with path.open('wb') as stream:
            proc = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        self.items.append((name, proc))
        print('Started:', name, '| log:', path, flush=True)

    def check(self):
        for name, proc in self.items:
            code = proc.poll()
            if code is not None:
                raise RuntimeError(f'{name} exited ({code}); see {self.directory / (name + ".log")}')

    def stop(self, names=None):
        chosen = [(n, p) for n, p in self.items if names is None or n in names]
        # Process groups belong exclusively to our start_new_session children.
        for sig, seconds in ((signal.SIGINT, 3), (signal.SIGTERM, 2), (signal.SIGKILL, 1)):
            remaining = []
            for name, proc in chosen:
                try:
                    os.killpg(proc.pid, sig)
                    remaining.append((name, proc))
                except ProcessLookupError:
                    pass
            if not remaining:
                break
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                for _, proc in remaining:
                    proc.poll()
                groups_alive = False
                for _, proc in remaining:
                    try:
                        os.killpg(proc.pid, 0)
                        groups_alive = True
                    except ProcessLookupError:
                        pass
                if not groups_alive:
                    break
                time.sleep(0.05)
            chosen = remaining
        for _, proc in chosen:
            proc.poll()


def make_probe():
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from nav_msgs.msg import Odometry, OccupancyGrid
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from ackermann_msgs.msg import AckermannDriveStamped

    class Probe(Node):
        def __init__(self):
            super().__init__('roboracer_session_check')
            self.seen = {}
            self.frames = {}
            self.drive_type = AckermannDriveStamped
            map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
            odom_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(OccupancyGrid, '/map', lambda m: self.receive('map', m), map_qos)
            self.create_subscription(Odometry, '/ego_racecar/odom', lambda m: self.receive('odom', m), odom_qos)
            self.create_subscription(PoseWithCovarianceStamped, '/localization/pose',
                                     lambda m: self.receive('pose', m), 10)

        def receive(self, key, msg):
            self.seen[key] = time.monotonic()
            self.frames[key] = msg.header.frame_id

        def fresh(self, key):
            return time.monotonic() - self.seen.get(key, float('-inf')) < 3

        def stopped_drive(self, rclpy):
            pub = self.create_publisher(self.drive_type, '/drive', 10)
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                pub.publish(self.drive_type())
                rclpy.spin_once(self, timeout_sec=0.05)
            self.destroy_publisher(pub)
    return Probe()


def runtime_paths():
    """Resolve only existing files; explicit overrides must never silently fall back."""
    adapter = Path(os.environ.get('ROBORACER_ADAPTER',
                                  str(Path(__file__).resolve().parent / 'sim_pose_adapter.py'))).expanduser().resolve()
    override = os.environ.get('ROBORACER_PLANNING_CONFIG')
    candidates = ([Path(override).expanduser()] if override else [
        Path.cwd() / 'src/planning/config/sim.yaml',
        Path.cwd() / '../f1tenth_ws/src/planning/config/sim.yaml',
    ])
    if not override and not any(p.is_file() for p in candidates):
        from ament_index_python.packages import get_package_share_directory
        candidates.append(Path(get_package_share_directory('planning')) / 'config/sim.yaml')
    config = next((p.resolve() for p in candidates if p.is_file()), None)
    ros = shutil.which('ros2')
    if not adapter.is_file():
        raise RuntimeError(f'Adapter not found: {adapter}')
    if config is None:
        raise RuntimeError('Planning config not found. Set ROBORACER_PLANNING_CONFIG to its full path.')
    if not ros:
        raise RuntimeError('ros2 not found after loading environment.')
    return adapter, config, ros


def preflight():
    # Imports and package discovery only: no nodes, children or drive messages.
    import importlib
    for name in ('rclpy', 'nav_msgs.msg', 'geometry_msgs.msg', 'ackermann_msgs.msg', 'f1tenth_gym'):
        importlib.import_module(name)
    from ament_index_python.packages import get_package_prefix
    for package, executable in (('f1tenth_gym_ros', 'gym_bridge'),
                                ('planning', 'occupancy_grid_planner'),
                                ('planning', 'pure_pursuit_follower'),
                                ('foxglove_bridge', 'foxglove_bridge')):
        path = Path(get_package_prefix(package)) / 'lib' / package / executable
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f'Missing executable: {path}; build/install the required package.')
    return runtime_paths()


def main():
    parser = argparse.ArgumentParser(description='One-terminal Ubuntu simulation session')
    parser.add_argument('--drive', action='store_true', help='also start low-speed follower (max 0.3 m/s)')
    parser.add_argument('--check', action='store_true', help='report existing processes/nodes; start nothing')
    parser.add_argument('--preflight', action='store_true', help='check installed environment; start no ROS nodes')
    args = parser.parse_args()
    if sum((args.drive, args.check, args.preflight)) > 1:
        parser.error('Choose only one of --drive, --check, --preflight.')
    try:
        adapter, config, ros_cli = preflight()
    except Exception as error:
        print('ERROR: Environment check failed:', error, flush=True)
        return 1
    if args.preflight:
        print('Environment check passed (runtime health not tested).')
        print('Workspace:', Path.cwd())
        print('Python:', sys.executable)
        print('Adapter:', adapter)
        print('Planning config:', config)
        print('ROS CLI:', ros_cli)
        return 0
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    probe = make_probe()
    children = None
    lock = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            rclpy.spin_once(probe, timeout_sec=0.1)
        processes = existing_processes()
        nodes = sorted(n for n, _ in probe.get_node_names_and_namespaces() if n in NODES)
        publishers = {topic: probe.count_publishers(topic) for topic in
                      ('/ego_racecar/odom', '/localization/pose', '/drive', '/cmd_vel', '/planner/path')}
        if args.check or processes or nodes or any(publishers.values()):
            print('Existing relevant processes:', processes)
            print('Existing relevant nodes:', nodes)
            print('Publisher counts:', publishers, flush=True)
            if args.check:
                return 0
            raise RuntimeError('Existing session detected. Stop old programs first; nothing was replaced.')
        state = Path.home() / '.local/state/roboracer-sim'
        state.mkdir(parents=True, exist_ok=True)
        lock = (state / 'session.lock').open('w')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another unified launcher is already running.')
        logdir = state / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + str(os.getpid()))
        logdir.mkdir()
        children = Children(logdir)
        ros = [sys.executable, ros_cli]
        def wait_for(label, predicate):
            end = time.monotonic() + 45
            while time.monotonic() < end:
                children.check()
                rclpy.spin_once(probe, timeout_sec=0.1)
                if predicate():
                    print('OK:', label, flush=True)
                    return
            raise RuntimeError('Timed out waiting for ' + label)
        children.start('simulation', ros + ['launch', 'f1tenth_gym_ros', 'gym_bridge_launch.py', 'open_foxglove:=false'])
        wait_for('map and live odometry', lambda: 'map' in probe.seen and probe.fresh('odom'))
        if probe.frames['map'] != 'map' or probe.frames['odom'] != 'map':
            raise RuntimeError('Expected map-frame map and odometry; refusing to relabel coordinates.')
        if probe.count_publishers('/localization/pose'):
            raise RuntimeError('Simulation already publishes localization; refusing duplicate adapter.')
        children.start('adapter', [sys.executable, str(adapter)])
        wait_for('live converted pose', lambda: probe.fresh('pose') and probe.count_publishers('/localization/pose') == 1)
        if probe.frames['pose'] != 'map':
            raise RuntimeError('Converted pose is not in map frame.')
        children.start('planner', ros + ['run', 'planning', 'occupancy_grid_planner'])
        wait_for('planner path publisher', lambda: probe.count_publishers('/planner/path') == 1)
        if args.drive:
            children.start('follower', ros + ['run', 'planning', 'pure_pursuit_follower', '--ros-args',
                           '--params-file', str(config), '-p', 'desired_speed:=0.3',
                           '-p', 'min_speed:=0.1', '-p', 'max_speed:=0.3'])
            wait_for('follower drive publisher', lambda: probe.count_publishers('/drive') == 1)
        print('\nREADY. Foxglove: Fixed frame = map; Display frame = map.', flush=True)
        print('2D pose topic: /planner/goal_pose. ' + ('Low-speed driving ENABLED.' if args.drive else 'Planning only; driving OFF.'), flush=True)
        print('Keep this terminal open. Control+C stops this session. Logs:', logdir, flush=True)
        while True:
            rclpy.spin_once(probe, timeout_sec=0.2)
            children.check()
            if not probe.fresh('odom') or not probe.fresh('pose'):
                raise RuntimeError('Odometry/localization missing for over 3 seconds; stopping session.')
            if probe.count_publishers('/localization/pose') != 1:
                raise RuntimeError('Localization publisher count changed; stopping session.')
    except KeyboardInterrupt:
        print('\nStopping this session...', flush=True)
        return 0
    except Exception as error:
        print('ERROR:', error, flush=True)
        return 1
    finally:
        # Do not let a second terminal signal interrupt cleanup halfway through.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        if children is not None:
            try:
                children.stop({'follower'})
                if any(n == 'simulation' for n, _ in children.items):
                    probe.stopped_drive(rclpy)
            finally:
                children.stop()
            print('Owned processes stopped. A zero-speed command was sent if simulation was started.', flush=True)
        probe.destroy_node()
        rclpy.shutdown()
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    sys.exit(main())
