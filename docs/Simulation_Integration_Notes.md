# RoboRacer Simulation Integration Notes

Updated: 2026-10-11

Record provenance: The file at the original linked location was no longer available. These notes were reconstructed from this conversation, user screenshots, and returned test results. They summarize verified outcomes and open work; they are not claimed to be a verbatim copy of the original file.

## Current environment and workflow

- User environment: Mac M4, VMware Fusion, Ubuntu 22.04 ARM64, ROS 2 Humble.
- Ubuntu workspace: `~/Desktop/roboracer-2025/sim_ws`.
- Current SSH address: `xinyi@172.16.4.128`; the IP may change with the network.
- Foxglove connection: `ws://172.16.4.128:8765`.
- Foxglove Fixed frame and Display frame are both set to `map`.
- Planning goals: use 2D pose to publish to `/planner/goal_pose`, with type PoseStamped.
- Initial-pose reset: `/initialpose`, with type PoseWithCovarianceStamped. This is separate from a planning goal.

## Converter responsibilities

Filename: `sim_pose_adapter.py`.

It subscribes to `/ego_racecar/odom` (Odometry) and copies the header and pose unchanged to `/localization/pose` (PoseWithCovarianceStamped), including the timestamp, coordinate frame, position, quaternion, and covariance.

It accepts only input already expressed in the `map` frame. It does not transform coordinates, copy velocity, estimate or improve localization accuracy, plan routes, or send driving commands.

## Verified results

### Converter data consistency during motion

At approximately 15:12 on 2026-10-11, the user ran `verify_pose_adapter.py` for about 45 seconds, set a goal during the test, and confirmed vehicle motion.

- Received odometry/localization messages: 11,261 / 11,270.
- Exact timestamp matches: 11,261 pairs.
- Moving samples (speed > 0.02 m/s): 3,211 pairs.
- Frame or value errors: 0.
- Maximum absolute differences for position, quaternion, and covariance: [0.0, 0.0, 0.0].
- Observed localization publisher count: 1.
- Maximum receive gaps: 0.070 / 0.072 seconds.
- Time since the last message at completion: 0.001 / 0.000 seconds.

Conclusion: Data-copy consistency passed for the samples in this run, including motion. The difference in message counts is not a direct packet-loss measurement. This check does not validate real-world localization accuracy, transport latency, long-term operation, or recovery after VM suspension.

### Single-terminal startup, motion, and shutdown

At 15:40–15:42 on 2026-10-11, the user tested the original unified launcher on Ubuntu:

1. After stopping the old programs, `--check` reported no relevant processes or nodes and zero monitored publishers.
2. With `--drive`, the simulation, adapter, planner, and follower started. All four readiness checks passed and READY appeared.
3. The user set a nearby straight-road goal in Foxglove and confirmed vehicle motion.
4. One Control+C in the same terminal triggered cleanup and returned to the shell prompt.
5. A separate `--check` reported processes `[]`, nodes `[]`, and zero publishers for `/ego_racecar/odom`, `/localization/pose`, `/drive`, `/cmd_vel`, and `/planner/path`.

Session log directory: `/home/xinyi/.local/state/roboracer-sim/20261011-194034-73320`.

Conclusion: Normal startup, goal-driven motion, and normal shutdown passed in this run. Cleanup reported sending zero-speed commands, but command receipt, vehicle velocity before exit, and final stopping error were not independently measured.

## Current daily use

Keep the Ubuntu VM running. After connecting to Ubuntu through SSH from a Mac terminal, run:

```bash
bash ~/Desktop/roboracer-2025/sim_ws/roboracer-start/run_sim.sh --drive
```

Wait for READY, then set a goal in Foxglove. To stop, press Control+C in that terminal.

Check session status with:

```bash
bash ~/Desktop/roboracer-2025/sim_ws/roboracer-start/run_sim.sh --check
```

Four separate terminals are no longer necessary. The check command itself does not start the simulator or follower.

## Open item 1: Planner frame validation and conversion — confirmed code issue

Ubuntu source: `~/Desktop/roboracer-2025/f1tenth_ws/src/planning/planning/occupancy_grid_planner.py`.

- `goal_callback()` uses the goal position x/y directly without validating or transforming the goal's header.frame_id.
- `pose_callback()` caches only the pose and discards the localization header. The localization frame also needs to be retained and checked.
- October 8 diagnostics showed map=map, pose=map, and goal=ego_racecar/base_link. Vehicle-relative coordinates were used directly as map coordinates.
- After changing Foxglove Display frame to map on October 11, diagnostics showed all three frames as map, the start and goal were connected, and the vehicle moved after the follower was started.

Required change: Validate both start and goal against the map message's coordinate frame. Transform differing frames using TF at an appropriate timestamp, or explicitly reject unsupported input. Merely changing the frame_id label is not a coordinate transformation. Missing frames or unavailable transforms must not lead to planning with untransformed coordinates.

Validation: Same-frame goals continue to work; vehicle-relative goals are correctly transformed or explicitly rejected; missing frames, unavailable TF, and localization-frame mismatches have clearly defined behavior.

Status: The original planner has not been modified. The current UI setting is a workaround, not a code fix.

## Open item 2: Unexpected initial-position and heading reset behavior — investigation needed

The user reported that setting an initial position/direction did not reset the vehicle as expected; it appeared to move forward instead.

Next steps: Confirm the selected publishing tool and topic; record the position/quaternion on `/initialpose` and the subsequent odometry; inspect simulator reset handling and whether driving commands were still active. Distinguish pose reset, goal following, and camera-view changes.

Status: The cause is unconfirmed and must not be attributed to the planner frame issue without evidence. No changes have been made.

## Open item 3: Apparent slowdown in the latter part of a route — measurement needed

The user visually suspected deceleration. No speed time series has been collected, and the follower's source-level deceleration logic has not been verified.

Next steps: Compare commanded speed on `/drive`, actual speed in odometry, and distance to the endpoint. Determine whether the behavior is expected, parameter-dependent, or a defect.

Status: Deferred. Normal goal-approach deceleration has not been established as the cause.

## Open item 4: Handling an old path after a new planning attempt fails

Observed source behavior: The planner returns when planning for a new goal fails, potentially leaving the previously retained path valid.

Next steps: Inspect both planner and follower and define how a failed plan invalidates the old path and stops following. Whether publishing an empty path is sufficient must be tested rather than assumed.

Status: No changes have been made. The existing planner/follower also do not provide dynamic obstacle avoidance based on live LiDAR. A grid route is not necessarily a feasible vehicle trajectory respecting turning-radius constraints.

## Open item 5: Deployment validation of the shared launcher

The shared package `roboracer-team-kit.zip` was generated during this conversation. A subsequent user screenshot confirmed that its extracted directory contained README.md, run_sim.py, run_sim.sh, sim_pose_adapter.py, and tests.

Changes in the shared version relative to the verified original:

- Includes the same converter.
- Defaults to using the launcher's parent directory as the workspace.
- Supports overrides for workspace, venv, ROS setup, adapter, and planning-config paths.
- Searches for configuration in source, the neighboring f1tenth_ws, or the installed share directory.
- Adds a `--preflight` environment check that creates no ROS nodes.
- Includes English instructions and a Chinese handoff note.

Four local test cases passed, covering path resolution and paths containing spaces, the original directory layout, rejection of a missing explicitly configured file, shell environment loading and missing-file diagnostics, selective/full child-process cleanup, and failure reporting. Shell/Python syntax, CLI help, and archive integrity checks also passed.

However, the new shared version has not completed runtime testing on Ubuntu or a teammate's computer. The original version's successful Ubuntu run must not be treated as validation of the shared revision. Recipients must first install and build compatible ROS, simulator, and planning packages; this package is not a complete system installer.

Recipient acceptance sequence: `--preflight` → `--check` → `--drive` → nearby straight-road goal and motion → Control+C → `--check` with no leftovers.

Untested: SSH-disconnection cleanup, VM suspension/recovery, message-interruption fault injection, and repeated restart cycles. A complete dependency lockfile and source commit records have not been collected. The shared package has not been committed or pushed to GitHub.

## Recommended next step

Address planner frame validation first, then define old-path invalidation and stopping behavior after planning failure. Investigate initial-pose and speed behavior separately, using measurements to establish their causes.
