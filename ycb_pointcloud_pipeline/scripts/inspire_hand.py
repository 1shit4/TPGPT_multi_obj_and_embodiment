#!/usr/bin/env python3
"""Inspire RH56 hand CLI without a built workspace.

The protocol, the fixed open/close grasp and the CLI live in the package module
``ycb_pointcloud_pipeline/inspire_hand.py``; this shim only makes it runnable
straight from the source tree::

    python3 scripts/inspire_hand.py scan
    python3 scripts/inspire_hand.py --speed 300 close

After ``colcon build`` the same CLI is ``ros2 run ycb_pointcloud_pipeline inspire_hand``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ycb_pointcloud_pipeline.inspire_hand import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
