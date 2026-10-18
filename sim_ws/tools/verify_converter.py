import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

REAL_N, REAL_INC = 1081, 0.004363323096185923
RUN_SECONDS = 8.0


def key(m):
    return (m.header.stamp.sec, m.header.stamp.nanosec)


class Verify(Node):
    def __init__(self):
        super().__init__('verify_converter')
        self.sim_scan, self.out_scan = {}, {}
        self.sim_odom, self.out_odom = {}, {}
        self.fail = {}
        self.count = {'scan': 0, 'odom': 0, 'pose': 0}
        self.prev = None
        self.create_subscription(LaserScan, '/sim/scan', lambda m: self.on(m, self.sim_scan, self.out_scan, self.chk_scan, 'scan', True), qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', lambda m: self.on(m, self.out_scan, self.sim_scan, self.chk_scan, 'scan', False), qos_profile_sensor_data)
        self.create_subscription(Odometry, '/sim/odom', lambda m: self.on(m, self.sim_odom, self.out_odom, self.chk_odom, 'odom', True), 10)
        self.create_subscription(Odometry, '/odom', lambda m: self.on(m, self.out_odom, self.sim_odom, self.chk_odom, 'odom', False), 10)
        self.create_timer(RUN_SECONDS, self.finish)

    def bad(self, name, detail):
        self.fail.setdefault(name, detail)   # keep the first example only

    def on(self, m, mine, other, chk, kind, is_sim):
        k = key(m)
        mine[k] = m
        if len(mine) > 300:
            mine.pop(next(iter(mine)))
        if k in other:
            sim, out = (m, other[k]) if is_sim else (other[k], m)
            self.count[kind] += 1
            chk(sim, out)

    def chk_scan(self, sim, out):
        s = np.asarray(sim.ranges, dtype=np.float32)
        if len(s) != REAL_N:
            s = s[np.round(np.linspace(0, len(s) - 1, REAL_N)).astype(int)]
        o = np.asarray(out.ranges, dtype=np.float32)
        if len(o) != REAL_N:
            self.bad('scan: beam count', f'{len(o)} != {REAL_N}')
        elif not np.allclose(s, o, equal_nan=True):
            self.bad('scan: ranges differ from sim', f'max diff {np.nanmax(np.abs(s - o))}')
        meta = {
            'frame_id': (out.header.frame_id, 'laser'),
            'angle_min': (out.angle_min, sim.angle_min),
            'angle_max': (out.angle_max, sim.angle_max),
            'angle_increment': (out.angle_increment, REAL_INC),
            'scan_time': (out.scan_time, 0.025),
            'range_min': (out.range_min, 0.02),
            'range_max': (out.range_max, 30.0),
        }
        for name, (got, want) in meta.items():
            same = got == want if isinstance(got, str) else math.isclose(got, want, rel_tol=1e-6)
            if not same:
                self.bad(f'scan: {name}', f'got {got}, want {want}')
        if out.time_increment <= 0.0:
            self.bad('scan: time_increment', 'still 0')
        if len(out.intensities) != 0:
            self.bad('scan: intensities', 'should be empty like the real car')

    def chk_odom(self, sim, out):
        v = sim.twist.twist.linear.x
        v = 0.0 if abs(v) < 0.05 else v
        if not math.isclose(out.twist.twist.linear.x, v, abs_tol=1e-9):
            self.bad('odom: linear.x', f'got {out.twist.twist.linear.x}, want {v}')
        if not math.isclose(out.twist.twist.angular.z, sim.twist.twist.angular.z, abs_tol=1e-9):
            self.bad('odom: angular.z', 'differs from sim')
        if out.twist.twist.linear.y != 0.0:
            self.bad('odom: linear.y', 'should be 0')
        if out.header.frame_id != 'odom' or out.child_frame_id != 'base_link':
            self.bad('odom: frames', f'{out.header.frame_id} -> {out.child_frame_id}')
        c = out.pose.covariance
        if (c[0], c[7], c[35]) != (0.2, 0.2, 0.4):
            self.bad('odom: pose covariance', f'{(c[0], c[7], c[35])}')
        q = out.pose.pose.orientation
        if abs(q.z ** 2 + q.w ** 2 - 1.0) > 1e-6 or q.x or q.y:
            self.bad('odom: orientation', 'not a unit yaw-only quaternion')

        # dead-reckoning: compare to integrating the converted twist ourselves
        p = out.pose.pose
        yaw = 2.0 * math.atan2(q.z, q.w)
        t = out.header.stamp.sec + out.header.stamp.nanosec * 1e-9
        if self.prev:
            px, py, pyaw, pt = self.prev
            dt = t - pt
            if 0.0 < dt < 1.0:
                self.count['pose'] += 1
                ex = px + out.twist.twist.linear.x * math.cos(pyaw) * dt
                ey = py + out.twist.twist.linear.x * math.sin(pyaw) * dt
                if abs(p.position.x - ex) > 1e-6 or abs(p.position.y - ey) > 1e-6:
                    self.bad('odom: pose integration', f'x {p.position.x} vs {ex}')
        self.prev = (p.position.x, p.position.y, yaw, t)

    def finish(self):
        print('\n=== converter verification ===')
        print(f"compared: {self.count['scan']} scans, {self.count['odom']} odoms, {self.count['pose']} pose steps")
        if self.count['scan'] == 0 or self.count['odom'] == 0:
            print('FAIL: no sim/converted pairs matched by stamp (is the launch file running?)')
        elif not self.fail:
            print('PASS: all checks')
        else:
            for name, detail in self.fail.items():
                print(f'FAIL  {name}: {detail}')
        raise SystemExit


def main():
    rclpy.init()
    try:
        rclpy.spin(Verify())
    except SystemExit:
        pass


if __name__ == '__main__':
    main()
