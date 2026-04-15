# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for the object lift environments.

Importing submodules here ensures Gym registrations in those modules run when the
package is imported (e.g., from task registry utilities).
"""

# Register default Franka tasks
# from . import franka  # noqa: F401
# Register custom Fanuc+panda finger tasks
from . import my_robot  # noqa: F401
# from . import fanuc
