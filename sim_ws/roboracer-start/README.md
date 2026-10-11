# RoboRacer simulation launcher — team handoff

A single foreground terminal starts the existing simulator, a pose-message adapter,
planner and (optionally) low-speed follower. Control+C cleans up this session.
This folder can be added to the team's repository under `sim_ws/roboracer-start/`.
It does not include or install the simulator, planner, ROS or their dependencies.

## Supported setup

The original launcher was tested on Ubuntu 22.04 ARM64 in VMware Fusion on an
Apple M4, with ROS 2 Humble. This portable revision has local path-resolution
and process-management tests; it still needs a first run on the recipient's Ubuntu.
Other architectures/ROS releases are not certified by these tests.

Before using it, prepare the team's working simulation workspace:

- ROS 2 Humble and a Python `.venv` created with `--system-site-packages` so it can
  import ROS modules. Install the simulator's Python dependencies in that venv.
- Built `f1tenth_gym_ros` and `planning` packages in `sim_ws/install`.
  The simulator used in the verified run is the upstream `dev-humble` variant,
  with `gym_bridge_launch.py` accepting `open_foxglove:=false` and starting the
  Foxglove bridge. A team-modified simulator may have different interfaces.
- Available executables: `gym_bridge`, `occupancy_grid_planner`,
  `pure_pursuit_follower`, and `foxglove_bridge` in their respective packages.
- Installed launch dependencies, including nav2 map server/lifecycle manager,
  xacro, robot/joint state publishers and Ackermann messages; use the project's
  environment/setup instructions to install and build them.
- A working simulator map/config and the team's `planning/config/sim.yaml`.
  Build Python ROS packages with the intended venv interpreter; the launcher
  cannot repair executables built against a different Python environment.

If starting from a fresh computer, complete the project's ROS/simulator setup
first. This is an add-on launcher, not a complete machine installer. Exact source
commit hashes and a full dependency lockfile were not captured in the original
run; identical reproduction on a new machine is not yet guaranteed.

## Install into an existing workspace

Extract the distribution archive, then copy the whole `roboracer-start` directory
into the simulation workspace. Keep the four runtime/documentation files together:

```text
<project>/
  sim_ws/
    .venv/
    install/local_setup.bash
    src/
    roboracer-start/
      run_sim.sh
      run_sim.py
      sim_pose_adapter.py
      README.md
  f1tenth_ws/src/planning/config/sim.yaml   (one supported config location)
```

Do not create a second nested `roboracer-start/roboracer-start` directory.
No executable permission change is needed when invoking the shell script with bash.
No username, home-directory name or VM IP is embedded in the startup command.

From **inside `sim_ws`**, run:

```bash
bash roboracer-start/run_sim.sh --preflight
```

This checks imports, executables and resolved paths without creating ROS nodes or
starting the simulation. It does not certify launch arguments, map contents or
live message flow. Fix any reported missing dependencies/paths before proceeding.

Then check for existing sessions:

```bash
bash roboracer-start/run_sim.sh --check
```

`--check` briefly creates an observer node and reports matching processes, nodes
and publisher counts. It starts no simulation or drive process and exits with 0
when the report completes, even if conflicts are present. A clear report has
empty process/node lists and all monitored publisher counts zero. Discovery can
take time; a single short report is not a guarantee of absence on a slow network.
Stop old sessions in their own terminals first, beginning with the follower.
Normal startup also checks conflicts and refuses to replace existing programs.

## Start and stop

Planning only (no follower):

```bash
bash roboracer-start/run_sim.sh
```

Simulation with the low-speed follower:

```bash
bash roboracer-start/run_sim.sh --drive
```

The follower uses `desired_speed=0.3`, `min_speed=0.1`, `max_speed=0.3` m/s.
Wait for `READY`; keep this terminal and the Ubuntu VM running.
To stop, press **Control+C** in this same terminal. The script stops its follower,
sends zero-speed commands while the simulator is still up, then stops its owned
process groups. Run `--check` afterward to check for leftovers. It does not kill
unrelated programs. Do not run a second driving controller in parallel.

For remote use, SSH into the Ubuntu machine first; the script runs on Ubuntu,
not in the Mac shell. Set Foxglove's WebSocket URL to `ws://<Ubuntu-IP>:8765`.
Use the current Ubuntu IP (`hostname -I`); no address is universally valid.

In Foxglove:

- Fixed frame: `map`; Display frame: `map`.
- 2D pose target topic: `/planner/goal_pose` (`PoseStamped`).
- Start with one nearby target in the same straight-road corridor.
- `/initialpose` is the separate initial-pose/reset topic, not a planning goal.

## What the converter does

The bundled `sim_pose_adapter.py` copies the odometry header and pose, including
orientation and covariance, from `/ego_racecar/odom` (`Odometry`) to
`/localization/pose` (`PoseWithCovarianceStamped`). Input must already use `map`.
It does not copy velocity, estimate localization, transform coordinates, plan
paths or issue driving commands. It was adapted from the team's supplied
Software Setup Guidance, Section 7; retain the team's applicable sharing terms.

This launcher expects the simulator NOT to publish `/localization/pose` itself.
If a team-modified simulator already publishes localization, startup refuses a
duplicate adapter. That setup requires adapting the launcher, not removing the
check blindly. Topics, frame names and the simulator launch entry are fixed to
the interfaces above; path overrides do not make other interfaces compatible.

## Different directory layouts

By default the script uses its parent directory as the simulation workspace,
`/opt/ros/humble/setup.bash` as ROS setup, and `<workspace>/.venv` as the venv.
It uses the bundled adapter; an old `sim_ws/sim_pose_adapter.py` is not selected.
Planning config is searched in this order:

1. `<workspace>/src/planning/config/sim.yaml`
2. `<workspace>/../f1tenth_ws/src/planning/config/sim.yaml`
3. The installed `planning` package's share directory, `config/sim.yaml`

Optional environment variables (use absolute paths):

- `ROBORACER_SIM_WS`: simulation workspace.
- `ROBORACER_VENV`: Python venv directory.
- `ROBORACER_ROS_SETUP`: ROS setup.bash file; Humble remains the tested release.
- `ROBORACER_PLANNING_CONFIG`: exact follower YAML configuration.
- `ROBORACER_ADAPTER`: optional replacement adapter Python file.

For example, when keeping the launcher outside the workspace:

```bash
ROBORACER_SIM_WS="$HOME/robot/sim_ws" bash /path/to/roboracer-start/run_sim.sh --preflight
```

Use the same overrides for subsequent `--check` and startup commands. A bad
explicit override causes an error rather than silently selecting another file.
ROS_DOMAIN_ID and middleware settings are inherited: keep them consistent with
your simulation and avoid sharing a ROS domain with another team's running car.

## Logs and troubleshooting

Logs: `~/.local/state/roboracer-sim/<timestamp>-<pid>/`, with a file per component.
The terminal prints the exact directory. If a child exits, inspect its log.

- Missing setup file: check workspace/venv path and build/install the project.
- Missing import or executable: complete the existing project environment setup.
- Existing session: stop the named old processes in their own terminals; retry.
- Timeout waiting for map/pose: inspect simulation/adapter logs, map config and
  frame names. Never fix a frame mismatch by only relabeling the header.
- Route visible but no movement: use `--drive`, check follower log and goal topic.
- Browser disconnects after Control+C: expected because the bridge was stopped.

The supervisor monitors process exits, localization publisher uniqueness and
odometry/localization freshness (3-second threshold). Cleanup on missing data
is a simulation convenience, not a validated physical-vehicle safety controller.
SSH-disconnection cleanup and VM suspension/recovery remain untested. A suspended
VM cannot execute cleanup until it runs again.

## Verification and open items

Original version, user-run Ubuntu test on 2026-10-11: readiness checks passed,
Foxglove target produced motion, Control+C cleanup completed, and the independent
post-stop check showed no matching processes/nodes or monitored publishers.
Converter test: 11,261 timestamp-matched pairs, including 3,211 moving pairs;
zero frame/value errors and one observed localization publisher.

Portable revision: path resolution, missing-environment diagnostics, CLI help,
syntax and child-process cleanup are tested locally. Repeat preflight, clean
startup, one straight-road goal, Control+C, and --check on the recipient's machine
before calling that deployment verified. See `tests/test_launcher.py` for local
checks; these do not simulate ROS or certify end-to-end operation.

Deferred separately: planner frame validation/TF conversion, initial-pose reset
behavior, and measured speed changes. Neither planner nor follower is modified
by this package. Failed planning may leave an old path active; the existing
planner/follower do not provide dynamic obstacle avoidance. Exact goal stopping
error and receipt of the shutdown zero-speed messages were not measured.
