"""
Scout Work Package
==================
Package pour la navigation autonome avec Velodyne LiDAR.

Modules:
- cloud: Gestion des PointCloud2
- ransac: Détection de murs via RANSAC
- planner: Planification de trajectoire et évitement d'obstacles
- markers: Visualisation avec RViz markers
"""

__version__ = "1.0.0"
__author__ = "Scout Team"

from . import cloud
from . import ransac
from . import planner
from . import markers

__all__ = ['cloud', 'ransac', 'planner', 'markers']