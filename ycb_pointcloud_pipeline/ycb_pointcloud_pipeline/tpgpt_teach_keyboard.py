#!/usr/bin/env python3
"""
Single-key control of tpgpt_teach_node, for the person teaching. Run it in its
own terminal (ros2 launch gives nodes no keyboard):

    ros2 run ycb_pointcloud_pipeline tpgpt_teach_keyboard
    ros2 run ycb_pointcloud_pipeline tpgpt_teach_keyboard --ros-args -p teach_node:=/tpgpt_teach_node

Keys
  c  capture_scene     object at its pick pose, arm OUT OF VIEW
  t  start_teaching    float the arm (asks for a typed 'yes')
  s  start_recording
  g  mark_grasp        CLOSE the gripper; while recording, the object is held from now on
  r  mark_release      OPEN the gripper; while recording, the object is let go from now on
  x  stop_recording    arm is held, demonstration converted, saved and plotted
  p  capture_placed    optional, after moving the arm out of view
  h  hold              switch to the hold controller now (discards a recording)
  q  quit              (does not change the controller)

Marks are stamped with the pose at the moment the node receives the key, so press
g with the fingers around the object and r where it should be let go, and hold the
arm still while the fingers move (gripper_mode:=none on the teach node: marks only).
"""

import select
import sys
import termios
import threading
import tty

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

KEYS = {
    'c': 'capture_scene', 't': 'start_teaching', 's': 'start_recording', 'g': 'mark_grasp',
    'r': 'mark_release', 'x': 'stop_recording', 'p': 'capture_placed', 'h': 'hold',
}
HELP = ('keys: c capture_scene | t start_teaching | s start_recording | g grasp (close) | '
        'r release (open) | x stop_recording | p capture_placed | h hold | q quit')


class TeachKeyboard(Node):
    def __init__(self):
        super().__init__('tpgpt_teach_keyboard')
        ns = self.declare_parameter('teach_node', '/tpgpt_teach_node').value.rstrip('/')
        self.clients_by_name = {name: self.create_client(Trigger, f'{ns}/{name}')
                                for name in KEYS.values()}

    def call(self, name):
        client = self.clients_by_name[name]
        if not client.service_is_ready():
            self.say(f'{client.srv_name} is not available -- is tpgpt_teach_node running?')
            return
        self.say(f'-> {name}')
        future = client.call_async(Trigger.Request())
        future.add_done_callback(lambda f, n=name: self._done(n, f))

    def _done(self, name, future):
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            self.say(f'<- {name} FAILED: {exc}')
            return
        self.say(f"<- {name} {'OK' if result.success else 'REFUSED'}: {result.message}")

    @staticmethod
    def say(text):
        # Raw terminal mode needs an explicit carriage return.
        sys.stdout.write(text.replace('\n', '\r\n') + '\r\n')
        sys.stdout.flush()


def main(args=None):
    rclpy.init(args=args)
    node = TeachKeyboard()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    if not sys.stdin.isatty():
        raise SystemExit('tpgpt_teach_keyboard needs an interactive terminal')
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    print(__doc__.split('Keys')[0].strip().splitlines()[0])
    print(HELP)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            if not select.select([sys.stdin], [], [], 0.2)[0]:
                continue
            key = sys.stdin.read(1).lower()
            if key == 'q':
                break
            if key == 't':
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)
                try:
                    answer = input("\nThe arm will FLOAT (zero stiffness). Support it, e-stop in "
                                   "reach. Type 'yes': ").strip().lower()
                finally:
                    tty.setcbreak(fd)
                if answer != 'yes':
                    node.say('cancelled')
                    continue
            if key in KEYS:
                node.call(KEYS[key])
            elif key.strip():
                node.say(HELP)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
