#!/usr/bin/env python3
"""
scout_work.py — Node principal pour navigation Velodyne
========================================================

Orchestre:
- Réception et filtrage des points LiDAR
- Détection des murs par RANSAC
- Planification de trajectoire avec évitement
- Publication des markers de visualisation

Topics:
- Sub:  /vlp16/points (PointCloud2)
- Pub:  /vlp16/points_filtered (PointCloud2 debug)
- Pub:  /scout/debug_markers (MarkerArray)
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

from .cloud import read_points_xyz_intensity_safe, make_cloud
from .ransac import ransac_line, remove_inliers, line_angle, wrap_pi
from .planner import PathPlanner
from .markers import MarkerPublisher


class ScoutWork(Node):
    """
    Node principal pour la navigation autonome avec LiDAR Velodyne.
    """
    
    def __init__(self):
        super().__init__('scout_work')

        # ============ Déclaration des paramètres ============
        
        # Topics
        self.declare_parameter('in_topic', '/vlp16/points')
        self.declare_parameter('out_topic', '/vlp16/points_filtered')
        self.declare_parameter('marker_topic', '/scout/debug_markers')

        # Filtering / ROI
        self.declare_parameter('max_range', 12.0)
        self.declare_parameter('min_z', -0.50)
        self.declare_parameter('max_z', 2.00)
        self.declare_parameter('roi_x_min', 0.25)
        self.declare_parameter('roi_x_max', 12.0)
        self.declare_parameter('roi_y_abs', 3.0)
        self.declare_parameter('voxel', 0.08)

        # RANSAC
        self.declare_parameter('ransac_iters', 260)
        self.declare_parameter('ransac_dist_th', 0.08)
        self.declare_parameter('ransac_min_inliers', 80)
        self.declare_parameter('use_theta_gate', True)
        self.declare_parameter('theta_gate_deg', 12.0)

        # Anti-flip + stabilization
        self.declare_parameter('x_ref_swap', 1.2)
        self.declare_parameter('swap_hysteresis', 0.05)
        self.declare_parameter('offset_lpf', 0.25)
        self.declare_parameter('offset_max_step', 0.10)

        # Centerline sampling
        self.declare_parameter('center_x_start', 0.6)
        self.declare_parameter('center_x_step', 0.5)
        self.declare_parameter('center_x_count', 12)

        # Centerline stabilization
        self.declare_parameter('center_lpf', 0.25)
        self.declare_parameter('center_hold_time', 0.6)

        # Safety corridor
        self.declare_parameter('robot_width', 0.70)
        self.declare_parameter('safety_margin', 0.35)

        # Obstacle avoidance
        self.declare_parameter('avoid_enable', True)
        self.declare_parameter('avoid_x_min', 0.4)
        self.declare_parameter('avoid_x_max', 3.0)
        self.declare_parameter('avoid_y_band', 1.2)
        self.declare_parameter('avoid_bias', 0.85)
        self.declare_parameter('avoid_apply_first_n', 6)
        self.declare_parameter('avoid_min_pts', 25)
        self.declare_parameter('avoid_z_min', 0.08)

        # Decision stability
        self.declare_parameter('obs_hyst', 0.12)
        self.declare_parameter('obs_hold_time', 0.8)

        # Wall exclusion
        self.declare_parameter('wall_key_eps', 0.03)

        # Marker config
        self.declare_parameter('marker_stamp_latest', True)

        # ============ Chargement des paramètres ============
        
        self.in_topic = str(self.get_parameter('in_topic').value)
        self.out_topic = str(self.get_parameter('out_topic').value)
        self.marker_topic = str(self.get_parameter('marker_topic').value)

        self.max_range = float(self.get_parameter('max_range').value)
        self.min_z = float(self.get_parameter('min_z').value)
        self.max_z = float(self.get_parameter('max_z').value)
        self.roi_x_min = float(self.get_parameter('roi_x_min').value)
        self.roi_x_max = float(self.get_parameter('roi_x_max').value)
        self.roi_y_abs = float(self.get_parameter('roi_y_abs').value)
        self.voxel = float(self.get_parameter('voxel').value)

        self.ransac_iters = int(self.get_parameter('ransac_iters').value)
        self.ransac_dist_th = float(self.get_parameter('ransac_dist_th').value)
        self.ransac_min_inliers = int(self.get_parameter('ransac_min_inliers').value)
        self.use_theta_gate = bool(self.get_parameter('use_theta_gate').value)
        self.theta_gate_deg = float(self.get_parameter('theta_gate_deg').value)

        self.avoid_enable = bool(self.get_parameter('avoid_enable').value)
        self.avoid_x_min = float(self.get_parameter('avoid_x_min').value)
        self.avoid_x_max = float(self.get_parameter('avoid_x_max').value)
        self.avoid_y_band = float(self.get_parameter('avoid_y_band').value)
        self.avoid_min_pts = int(self.get_parameter('avoid_min_pts').value)
        self.avoid_z_min = float(self.get_parameter('avoid_z_min').value)

        self.wall_key_eps = float(self.get_parameter('wall_key_eps').value)

        # ============ Initialisation des modules ============
        
        # Planificateur de trajectoire
        self.planner = PathPlanner(self)
        params = {
            'x_ref_swap': float(self.get_parameter('x_ref_swap').value),
            'swap_hysteresis': float(self.get_parameter('swap_hysteresis').value),
            'offset_lpf': float(self.get_parameter('offset_lpf').value),
            'offset_max_step': float(self.get_parameter('offset_max_step').value),
            'center_x_start': float(self.get_parameter('center_x_start').value),
            'center_x_step': float(self.get_parameter('center_x_step').value),
            'center_x_count': int(self.get_parameter('center_x_count').value),
            'center_lpf': float(self.get_parameter('center_lpf').value),
            'center_hold_time': float(self.get_parameter('center_hold_time').value),
            'robot_width': float(self.get_parameter('robot_width').value),
            'safety_margin': float(self.get_parameter('safety_margin').value),
            'avoid_enable': self.avoid_enable,
            'avoid_bias': float(self.get_parameter('avoid_bias').value),
            'avoid_apply_first_n': int(self.get_parameter('avoid_apply_first_n').value),
            'obs_hyst': float(self.get_parameter('obs_hyst').value),
            'obs_hold_time': float(self.get_parameter('obs_hold_time').value),
            'roi_x_min': self.roi_x_min,
            'roi_x_max': self.roi_x_max,
        }
        self.planner.load_params(params)

        # Publisher de markers
        self.marker_pub = MarkerPublisher(self, self.marker_topic)
        marker_params = {
            'marker_stamp_latest': bool(self.get_parameter('marker_stamp_latest').value),
            'roi_x_min': self.roi_x_min,
            'roi_x_max': self.roi_x_max,
        }
        self.marker_pub.set_params(marker_params)

        # ============ Publishers & Subscribers ============
        
        self.pub_cloud = self.create_publisher(PointCloud2, self.out_topic, 10)
        self.sub = self.create_subscription(PointCloud2, self.in_topic, self.cb, 10)

        # ============ État interne ============
        
        self.theta_prev = None
        self._printed_fields = False

        # ============ Log de démarrage ============
        
        self.get_logger().info(
            f"[ScoutWork] Initialized\n"
            f"  Topics: in={self.in_topic} out={self.out_topic} markers={self.marker_topic}\n"
            f"  ROI: x=[{self.roi_x_min},{self.roi_x_max}] y=±{self.roi_y_abs} "
            f"z=[{self.min_z},{self.max_z}] voxel={self.voxel}\n"
            f"  RANSAC: iters={self.ransac_iters} dist_th={self.ransac_dist_th} "
            f"min_inl={self.ransac_min_inliers} theta_gate={self.use_theta_gate}\n"
            f"  Avoid: enable={self.avoid_enable}"
        )

    def _key_xy(self, x, y):
        """Crée une clé spatiale pour comparaison de points."""
        e = self.wall_key_eps
        return (int(x / e), int(y / e))

    def cb(self, msg: PointCloud2):
        """
        Callback principal sur réception d'un scan LiDAR.
        
        Args:
            msg: Message PointCloud2
        """
        # Log des champs une seule fois
        if not self._printed_fields:
            self._printed_fields = True
            fields_str = ", ".join([f"{f.name}@{f.offset}" for f in msg.fields])
            self.get_logger().info(f"PointCloud2 fields: {fields_str}")

        # ============ 1. Lecture et filtrage ============
        
        pts = read_points_xyz_intensity_safe(msg)

        filtered = []
        for x, y, z, it in pts:
            # Validité des valeurs
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            # Range max
            if math.hypot(x, y) > self.max_range:
                continue
            # Filtre Z
            if z < self.min_z or z > self.max_z:
                continue
            # ROI X
            if x < self.roi_x_min or x > self.roi_x_max:
                continue
            # ROI Y
            if abs(y) > self.roi_y_abs:
                continue
            filtered.append((x, y, z, it))

        # ============ 2. Voxel downsampling ============
        
        if self.voxel > 0.0 and filtered:
            vox = {}
            inv = 1.0 / self.voxel
            for x, y, z, it in filtered:
                k = (int(x * inv), int(y * inv), int(z * inv))
                if k not in vox:
                    vox[k] = (x, y, z, it)
            filtered = list(vox.values())

        # Publication du nuage filtré
        self.pub_cloud.publish(make_cloud(filtered, msg.header.frame_id, msg.header.stamp))

        # ============ 3. Extraction XY pour RANSAC ============
        
        points_xy = [(x, y) for (x, y, _z, _it) in filtered]
        
        if len(points_xy) < 120:
            self.marker_pub.publish_markers(
                msg.header, None, None, [], 0, 0,
                obstacle_side=self.planner.obs_side_mem,
                pts_count=len(points_xy)
            )
            return

        # ============ 4. RANSAC #1 (premier mur) ============
        
        theta_hint = None
        if self.use_theta_gate and self.theta_prev is not None:
            theta_hint = self.theta_prev

        line1, in1 = ransac_line(
            points_xy,
            iters=self.ransac_iters,
            dist_th=self.ransac_dist_th,
            min_inliers=self.ransac_min_inliers,
            theta_hint=theta_hint,
            theta_max_dev_deg=self.theta_gate_deg
        )

        if line1 is None:
            centers = self.planner.hold_centers_if_needed([])
            self.marker_pub.publish_markers(
                msg.header, None, None, centers, 0, 0,
                obstacle_side=self.planner.obs_side_mem,
                pts_count=len(points_xy)
            )
            return

        # ============ 5. RANSAC #2 (second mur) ============
        
        rest = remove_inliers(points_xy, in1, eps=1e-4)
        line2, in2 = ransac_line(
            rest,
            iters=self.ransac_iters,
            dist_th=self.ransac_dist_th,
            min_inliers=self.ransac_min_inliers,
            theta_hint=theta_hint,
            theta_max_dev_deg=self.theta_gate_deg
        )

        if line2 is None:
            self.theta_prev = line_angle(line1)
            centers = self.planner.hold_centers_if_needed([])
            self.marker_pub.publish_markers(
                msg.header, line1, None, centers, len(in1), 0,
                obstacle_side=self.planner.obs_side_mem,
                pts_count=len(points_xy)
            )
            return

        # ============ 6. Assignation initiale gauche/droite ============
        
        mean1 = sum(p[1] for p in in1) / len(in1)
        mean2 = sum(p[1] for p in in2) / len(in2)

        if mean1 >= mean2:
            left_line, left_in = line1, in1
            right_line, right_in = line2, in2
        else:
            left_line, left_in = line2, in2
            right_line, right_in = line1, in1

        # ============ 7. Stabilisation des murs ============
        
        left_line, right_line, left_in, right_in = self.planner.stabilize_walls(
            left_line, right_line, left_in, right_in
        )

        # ============ 8. Mise à jour theta_prev ============
        
        thL = line_angle(left_line)
        thR = line_angle(right_line)
        dth = wrap_pi(thR - thL)
        self.theta_prev = wrap_pi(thL + 0.5 * dth)

        # ============ 9. Calcul centerline ============
        
        centers = self.planner.compute_centerline(left_line, right_line)

        # ============ 10. Détection obstacles ============
        
        # Construction des clés spatiales des murs
        wall_keys = set()
        for px, py in left_in:
            wall_keys.add(self._key_xy(px, py))
        for px, py in right_in:
            wall_keys.add(self._key_xy(px, py))

        # Extraction des obstacles (hors murs)
        obs = []
        if self.avoid_enable and centers:
            for (x, y, z, _it) in filtered:
                if x < self.avoid_x_min or x > self.avoid_x_max:
                    continue
                if abs(y) > self.avoid_y_band:
                    continue
                if z < self.avoid_z_min:
                    continue
                if self._key_xy(x, y) in wall_keys:
                    continue
                obs.append((x, y))

        # ============ 11. Évitement obstacles ============
        
        obstacle_side = self.planner.obs_side_mem
        if len(obs) >= self.avoid_min_pts:
            xref = max(self.roi_x_min + 0.2, 
                      min(self.roi_x_max - 0.2, self.planner.x_ref_swap))
            centers, obstacle_side = self.planner.apply_obstacle_avoidance(
                centers, obs, wall_keys, xref
            )

        # ============ 12. Lissage et hold ============
        
        centers = self.planner.smooth_centers(centers)
        centers = self.planner.hold_centers_if_needed(centers)

        # ============ 13. Publication markers ============
        
        self.marker_pub.publish_markers(
            msg.header, left_line, right_line, centers,
            len(left_in), len(right_in),
            obstacle_side=obstacle_side,
            pts_count=len(points_xy)
        )


def main():
    """Point d'entrée principal."""
    rclpy.init()
    node = ScoutWork()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()