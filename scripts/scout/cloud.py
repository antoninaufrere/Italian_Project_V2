#!/usr/bin/env python3
"""
cloud.py - Gestion des PointCloud2
===================================

Fonctions pour lire et créer des messages PointCloud2 ROS2.
Supporte les champs x, y, z, intensity avec lecture robuste.
Inclut la projection 2D pour le traitement RANSAC.
"""

import struct
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


# Constants pour les types de données PointCloud2
INT8 = 1
UINT8 = 2
INT16 = 3
UINT16 = 4
INT32 = 5
UINT32 = 6
FLOAT32 = 7
FLOAT64 = 8

_DATATYPE_STRUCT = {
    INT8: ('b', 1),
    UINT8: ('B', 1),
    INT16: ('h', 2),
    UINT16: ('H', 2),
    INT32: ('i', 4),
    UINT32: ('I', 4),
    FLOAT32: ('f', 4),
    FLOAT64: ('d', 8),
}


def _field_map(msg: PointCloud2):
    """Crée un dictionnaire des champs avec leurs offsets et types."""
    fmap = {}
    for f in msg.fields:
        fmap[f.name] = (f.offset, f.datatype)
    return fmap


def _read_one(data: bytes, base: int, offset: int, datatype: int):
    """Lit une valeur à partir des données binaires."""
    fmt, _size = _DATATYPE_STRUCT[datatype]
    return struct.unpack_from('<' + fmt, data, base + offset)[0]


def read_points_xyz_intensity_safe(msg: PointCloud2):
    """
    Lit les points x, y, z, intensity depuis un PointCloud2.
    
    Args:
        msg: Message PointCloud2
        
    Returns:
        Liste de tuples (x, y, z, intensity). intensity=0.0 si absent.
    """
    fmap = _field_map(msg)
    if 'x' not in fmap or 'y' not in fmap or 'z' not in fmap:
        return []

    ox, tx = fmap['x']
    oy, ty = fmap['y']
    oz, tz = fmap['z']
    has_i = 'intensity' in fmap
    if has_i:
        oi, ti = fmap['intensity']

    step = msg.point_step
    data = msg.data
    n = msg.width * msg.height

    out = []
    for i in range(n):
        base = i * step
        if base + step > len(data):
            break
        x = float(_read_one(data, base, ox, tx))
        y = float(_read_one(data, base, oy, ty))
        z = float(_read_one(data, base, oz, tz))
        it = float(_read_one(data, base, oi, ti)) if has_i else 0.0
        out.append((x, y, z, it))
    return out


def project_to_2d(points_3d):
    """
    Projette les points 3D sur le plan XY (vue de dessus).
    Élimine la coordonnée Z pour le traitement RANSAC des lignes.
    
    Args:
        points_3d: Liste de tuples (x, y, z, intensity)
        
    Returns:
        Liste de tuples (x, y) - projection 2D
    """
    return [(x, y) for (x, y, z, it) in points_3d]


def project_to_2d_with_intensity(points_3d):
    """
    Projette les points 3D sur le plan XY en conservant l'intensité.
    
    Args:
        points_3d: Liste de tuples (x, y, z, intensity)
        
    Returns:
        Liste de tuples (x, y, intensity) - projection 2D avec intensité
    """
    return [(x, y, it) for (x, y, z, it) in points_3d]


def project_to_2d_keep_z(points_3d):
    """
    Projette les points 3D sur le plan XY en conservant Z comme métadonnée.
    Utile pour le filtrage d'obstacles par hauteur après projection.
    
    Args:
        points_3d: Liste de tuples (x, y, z, intensity)
        
    Returns:
        Tuple (points_2d, z_values):
            - points_2d: Liste de tuples (x, y)
            - z_values: Liste des valeurs Z correspondantes
    """
    points_2d = [(x, y) for (x, y, z, it) in points_3d]
    z_values = [z for (x, y, z, it) in points_3d]
    return points_2d, z_values


def make_cloud(points, frame_id: str, stamp):
    """
    Crée un message PointCloud2 à partir d'une liste de points.
    
    Args:
        points: Liste de tuples (x, y, z, intensity)
        frame_id: Frame de référence
        stamp: Timestamp du message
        
    Returns:
        Message PointCloud2 formaté
    """
    fields = [
        PointField(name='x', offset=0, datatype=FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=FLOAT32, count=1),
    ]
    buf = bytearray()
    for (x, y, z, it) in points:
        buf += struct.pack('<ffff', float(x), float(y), float(z), float(it))

    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = len(points)
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.data = bytes(buf)
    msg.is_dense = True
    return msg


def make_cloud_2d(points_2d, frame_id: str, stamp, z_default=0.0):
    """
    Crée un message PointCloud2 à partir de points 2D.
    Les points sont placés sur le plan Z=z_default.
    
    Args:
        points_2d: Liste de tuples (x, y)
        frame_id: Frame de référence
        stamp: Timestamp du message
        z_default: Valeur Z par défaut (défaut: 0.0)
        
    Returns:
        Message PointCloud2 formaté
    """
    points_3d = [(x, y, z_default, 0.0) for (x, y) in points_2d]
    return make_cloud(points_3d, frame_id, stamp)