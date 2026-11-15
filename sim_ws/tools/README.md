# Real-robot message format workflow

These tools adapt live simulator scans and odometry to resemble the real
robot's ROS messages. They do not replay a recorded drive or calibrate the
simulator's physical model.

## Run in a configured ROS 2 environment

From `sim_ws`, start the simulator and converter:

```bash
source install/setup.bash
ros2 launch tools/real_format.launch.py
```

The launch wrapper routes raw simulator messages to `/sim/scan` and
`/sim/odom`. The converter publishes `/scan`, `/odom`, and transforms for
`odom -> base_link -> laser`.

In a second terminal, also from `sim_ws`:

```bash
source install/setup.bash
python3 tools/verify_converter.py
```

This checks timestamp-matched simulator and converted messages for eight
seconds. It checks selected metadata, scan values, velocity conversion, and
position integration. It does not establish physical accuracy, validate all
transforms, or compare the simulated trajectory against a real drive.

To compare selected message-format fields with a real recording:

```bash
python3 tools/compare_to_real.py /path/to/real_ref
```

The recording must contain `/scan` and `/odom` in a SQLite3 ROS bag. It is
supplied separately and is not included in this contribution. The comparison
uses the first recorded and first received message for each topic; it does
not compare complete time series or actual trajectories. Read the printed
results: the tools do not currently return a nonzero exit status for failed
checks.

## Relationship to the planning launcher

Use this as a separate session from `roboracer-start/run_sim.sh`. Both start
the simulator, so do not run them together.

The planning launcher expects raw map-frame odometry on `/ego_racecar/odom`
and copies its pose to `/localization/pose`. This workflow remaps that raw
topic to `/sim/odom`. Its converted `/odom` starts at a local origin and is
integrated from velocity; it cannot directly replace map-frame localization.

Integration with the planning launcher remains untested. A combined workflow
would need explicit topic routing so the pose adapter receives raw map-frame
odometry while the converter publishes its separate real-format output.

## Validation limits

Selected reference-format fields passed in the author's recorded comparison.
Combined ROS runtime behavior after merging the planning launcher has not
been tested. Scan timing fields are assigned without enforcing the stated
publication rate. The converter advertises a 0.02–30 m range while the
current simulator configuration uses 0.05–25 m; changing metadata does not
extend the simulated sensor's measurement capability.
