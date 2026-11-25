# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for SLURM configuration and cluster settings.
"""

from enum import Enum
import platform
import typing as tp

import omegaconf


class ClusterType(Enum):
    LOCAL_DARWIN = "darwin"
    DEFAULT = "default"  # used for any other cluster.


def _guess_cluster_type() -> ClusterType:
    if platform.system() == "Darwin":
        return ClusterType.LOCAL_DARWIN

    return ClusterType.DEFAULT


def get_cluster_type(
    cluster_type: tp.Optional[ClusterType] = None,
) -> tp.Optional[ClusterType]:
    if cluster_type is None:
        return _guess_cluster_type()

    return cluster_type


def get_slurm_parameters(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Apply the optional exclusion list from the user's configuration."""
    from ..environment import AudioCraftEnvironment
    slurm_exclude = AudioCraftEnvironment.get_slurm_exclude()
    if slurm_exclude is not None:
        cfg["exclude"] = slurm_exclude
    return cfg
