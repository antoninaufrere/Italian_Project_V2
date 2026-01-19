#!/usr/bin/env python3
"""
go_to_pose.py - Navigation en serpentin dans les vignes
========================================================

États principaux:
1. GO_TO_POSE classique (aller au début du 1er rang)
2. FOLLOW_LINE (suivre le chemin central)
3. EXIT_FORWARD (sortir du rang)
4. TURN_90_RIGHT (pivoter à droite - 180° pour retour OU 90° pour changement rang)
5. SCAN_FOR_NEW_ROWS (chercher les nouveaux murs avec LiDAR)
6. MOVE_TO_CENTER (se centrer entre les nouveaux murs)
7. TURN_90_RIGHT (encore 90° à droite pour finir le demi-tour = 180° total)
8. Retour en FOLLOW_LINE

Mode serpentin activable via paramètre.
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String


def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


def wrap_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def yaw_from_quat(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


class GoToPose(Node):
    def __init__(self):
        super().__init__('go_to_pose')

        # Publisher pour l'état de navigation
        self.state_pub = self.create_publisher(
            String, 
            '/scout/navigation_state', 
            10
        )
        
        # État actuel
        self.current_state = "IDLE"
        
        # Timer pour publier l'état régulièrement
        self.state_timer = self.create_timer(0.1, self.publish_state)
        # -------- Params GO TO POSE --------
        self.declare_parameter('goal_x', -1.0)
        self.declare_parameter('goal_y', -2.0)
        self.declare_parameter('goal_yaw', 0.0)

        self.goal_x = float(self.get_parameter('goal_x').value)
        self.goal_y = float(self.get_parameter('goal_y').value)
        self.goal_yaw = float(self.get_parameter('goal_yaw').value)

        # -------- Params FOLLOW LINE (scout_work) --------
        self.declare_parameter('marker_topic', '/scout/debug_markers')
        self.declare_parameter('line_ns', 'center_path')
        self.declare_parameter('follow_v', 0.50)
        self.declare_parameter('follow_k', 2.0)
        self.declare_parameter('lookahead', 1.5)
        self.declare_parameter('line_timeout', 0.3)
        self.declare_parameter('max_w_follow', 1.2)

        self.marker_topic = str(self.get_parameter('marker_topic').value)
        self.line_ns = str(self.get_parameter('line_ns').value)
        self.follow_v = float(self.get_parameter('follow_v').value)
        self.follow_k = float(self.get_parameter('follow_k').value)
        self.lookahead = float(self.get_parameter('lookahead').value)
        self.line_timeout = float(self.get_parameter('line_timeout').value)
        self.max_w_follow = float(self.get_parameter('max_w_follow').value)

        # -------- Params WAIT LINE --------
        self.declare_parameter('wait_line_time', 4.0)
        self.wait_line_time = float(self.get_parameter('wait_line_time').value)
        self.wait_line_deadline = None

        # -------- Params EXIT FORWARD --------
        self.declare_parameter('exit_distance', 1.0)
        self.declare_parameter('exit_v', 0.35)
        self.declare_parameter('exit_w_max', 0.8)

        self.exit_distance = float(self.get_parameter('exit_distance').value)
        self.exit_v = float(self.get_parameter('exit_v').value)
        self.exit_w_max = float(self.get_parameter('exit_w_max').value)

        # -------- Params SERPENTIN MODE --------
        self.declare_parameter('serpentin_mode', True)
        self.declare_parameter('max_rows', 10)
        self.declare_parameter('inter_row_distance', 2.0)  # Distance entre rangs (m)
        self.declare_parameter('rotation_direction', 1)  # -1=droite (Y+), +1=gauche (Y-)
        
        self.serpentin_mode = bool(self.get_parameter('serpentin_mode').value)
        self.max_rows = int(self.get_parameter('max_rows').value)
        self.inter_row_distance = float(self.get_parameter('inter_row_distance').value)
        self.rotation_direction = int(self.get_parameter('rotation_direction').value)

        # -------- Params TURN 90° --------
        self.declare_parameter('turn_90_tol', 0.08)
        self.declare_parameter('turn_90_k', 2.0)
        self.declare_parameter('turn_90_max_w', 1.0)
        self.declare_parameter('turn_90_min_w', 0.18)

        self.turn_90_tol = float(self.get_parameter('turn_90_tol').value)
        self.turn_90_k = float(self.get_parameter('turn_90_k').value)
        self.turn_90_max_w = float(self.get_parameter('turn_90_max_w').value)
        self.turn_90_min_w = float(self.get_parameter('turn_90_min_w').value)

        # -------- Params SCAN FOR NEW ROWS --------
        self.declare_parameter('scan_timeout', 3.0)
        self.declare_parameter('min_wall_y', 0.5)
        
        self.scan_timeout = float(self.get_parameter('scan_timeout').value)
        self.min_wall_y = float(self.get_parameter('min_wall_y').value)

        # -------- Params MOVE TO CENTER --------
        self.declare_parameter('center_tolerance', 0.15)
        self.declare_parameter('center_v', 0.30)
        self.declare_parameter('center_k_lateral', 1.5)
        
        self.center_tolerance = float(self.get_parameter('center_tolerance').value)
        self.center_v = float(self.get_parameter('center_v').value)
        self.center_k_lateral = float(self.get_parameter('center_k_lateral').value)

        # -------- Gains / limits GO TO POSE --------
        self.k_yaw = 2.5
        self.k_yaw_drive = 2.5
        self.k_lin = 0.8
        self.max_w = 1.5
        self.max_v = 0.6
        self.min_w = 0.15
        self.k_yaw_final = 1.6
        self.max_w_final = 0.8
        self.yaw_tol = 0.05
        self.pos_tol = 0.05
        self.final_yaw_tol = 0.05
        self.yaw_back_to_turn = 1.8 * self.yaw_tol

        # -------- State --------
        self.state = "WAIT_ODOM"
        self.have_odom = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.ignore_markers = False  # Flag pour ignorer les markers pendant rotations
        self.get_logger().info("Rotation terminée -> Réactivation markers")


        # -------- Line tracking --------
        self.line_points = []
        self.last_line_time = None

        # -------- Wall tracking (from scout_work markers) --------
        self.left_wall_y = None
        self.right_wall_y = None
        self.last_walls_time = None
        self.walls_timeout = 0.5

        # -------- Exit-forward --------
        self.last_follow_w = 0.0
        self.exit_end_time = None

        # -------- Turn 90° --------
        self.turn_target_yaw = None

        # -------- Scan for new rows --------
        self.scan_deadline = None
        self.new_left_wall_y = None
        self.new_right_wall_y = None

        # -------- Move lateral (changement de rang) --------
        self.lateral_start_x = None
        self.lateral_start_y = None
        self.lateral_target_distance = None

        # -------- Row counter --------
        self.rows_completed = 0
        self.half_turns_done = 0
        self.row_change_step = 0  # 0=pas en changement, 1=align, 2=1er 90°, 3=scan+avance, 4=2e 90°
        self.row_exit_yaw = None  # Yaw de sortie du rang (pour réalignement)

        # -------- Pubs/Subs --------
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(MarkerArray, self.marker_topic, self.marker_cb, 10)

        self.timer = self.create_timer(0.02, self.loop)

        self.get_logger().info(
            f"[GoToPose] Goal: x={self.goal_x:.2f} y={self.goal_y:.2f} yaw={self.goal_yaw:.2f}"
        )
        self.get_logger().info(
            f"[GoToPose] Serpentin: {self.serpentin_mode} | Max rows: {self.max_rows}"
        )

    # ================ CALLBACKS ================

    def publish_state(self):
        """Publie l'état actuel pour scout_work"""
        msg = String()
        
        # Map internal states to scout_work expected states
        if self.state == "FOLLOW_LINE":
            msg.data = "FOLLOWING_ROW"
        elif self.ignore_markers:
            msg.data = "TURNING_180"
        elif self.state in ["TURN_90_RIGHT", "TURN_90_LEFT", "ALIGN_FOR_TURN"]:
            msg.data = "TURNING_180"
        elif self.state in ["EXIT_FORWARD", "WAIT_LINE"]:
            msg.data = "END_OF_ROW"
        elif self.state in ["SCAN_FOR_NEW_ROWS", "MOVE_LATERAL", "MOVE_TO_CENTER", "MOVE_TO_CENTERLINE"]:
            msg.data = "ALIGNING"
        else:
            msg.data = "IDLE"
        
        self.state_pub.publish(msg)

    def odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.have_odom = True

        if self.state == "WAIT_ODOM":
            self.state = "TURN_TO_GOAL"
            self.stop()

    def marker_cb(self, msg: MarkerArray):
        """Récupère la ligne centrale ET les positions des murs."""
        self.get_logger().info(f"[DEBUG] marker_cb called | ignore_markers={self.ignore_markers}")

        # Ignorer les markers pendant les rotations
        if self.ignore_markers:
            return
            
        for m in msg.markers:
            # Ligne centrale
            if m.ns == self.line_ns and m.type == m.LINE_STRIP:
                pts = [(float(p.x), float(p.y)) for p in m.points]
                self.line_points = pts
                self.last_line_time = self.get_clock().now()
            
            # Mur gauche
            elif m.ns == "wall_left" and m.type == m.LINE_STRIP and len(m.points) >= 2:
                target_x = 1.5
                best_pt = None
                best_err = 1e9
                for p in m.points:
                    if p.x > 0.3:
                        err = abs(p.x - target_x)
                        if err < best_err:
                            best_err = err
                            best_pt = p
                if best_pt:
                    self.left_wall_y = float(best_pt.y)
                    self.last_walls_time = self.get_clock().now()
            
            # Mur droit
            elif m.ns == "wall_right" and m.type == m.LINE_STRIP and len(m.points) >= 2:
                target_x = 1.5
                best_pt = None
                best_err = 1e9
                for p in m.points:
                    if p.x > 0.3:
                        err = abs(p.x - target_x)
                        if err < best_err:
                            best_err = err
                            best_pt = p
                if best_pt:
                    self.right_wall_y = float(best_pt.y)
                    self.last_walls_time = self.get_clock().now()

    # ================ HELPERS ================

    def publish_cmd(self, v, w):
        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(w)
        self.cmd_pub.publish(t)

    def stop(self):
        self.publish_cmd(0.0, 0.0)

    def apply_min_w(self, w, min_w):
        if min_w <= 0.0:
            return w
        if abs(w) < 1e-6:
            return w
        if abs(w) < min_w:
            return math.copysign(min_w, w)
        return w

    def line_is_fresh(self):
        if self.last_line_time is None:
            return False
        age = (self.get_clock().now() - self.last_line_time).nanoseconds * 1e-9
        return age <= self.line_timeout

    def walls_are_fresh(self):
        if self.last_walls_time is None:
            return False
        age = (self.get_clock().now() - self.last_walls_time).nanoseconds * 1e-9
        return age <= self.walls_timeout

    def pick_target_point(self):
        if not self.line_points:
            return None

        best = None
        best_err = 1e9
        L = max(0.2, self.lookahead)

        for (x, y) in self.line_points:
            if x <= 0.05:
                continue
            d = math.hypot(x, y)
            err = abs(d - L)
            if err < best_err:
                best_err = err
                best = (x, y)
        return best

    # ================ STATE TRANSITIONS ================

    def start_wait_line(self):
        if self.state == "WAIT_LINE":
            return
        self.get_logger().warn("Ligne perdue -> WAIT_LINE")
        self.state = "WAIT_LINE"
        self.wait_line_deadline = self.get_clock().now() + Duration(seconds=self.wait_line_time)
        self.stop()

    def start_exit_forward(self):
        if self.state == "EXIT_FORWARD":
            return
        self.get_logger().warn("Ligne absente -> EXIT_FORWARD")
        self.state = "EXIT_FORWARD"
        t = max(0.1, self.exit_distance / max(0.05, self.exit_v))
        self.exit_end_time = self.get_clock().now() + Duration(seconds=t)
        # Capturer le yaw actuel pour réalignement ultérieur
        self.row_exit_yaw = self.yaw

    def start_turn_90_right(self):
        """Tourne de -90° (à droite). Peut être appelé 2 fois pour faire -180°."""
        self.turn_target_yaw = wrap_pi(self.yaw - math.pi / 2.0)
        self.get_logger().info(f"TURN -90° (step={self.row_change_step}) de {self.yaw:.2f} vers {self.turn_target_yaw:.2f}")
        self.state = "TURN_90_RIGHT"
        self.stop()

    def start_turn_90_left(self):
        self.turn_target_yaw = wrap_pi(self.yaw - math.pi / 2.0)
        self.get_logger().info(f"TURN_90_LEFT -> {self.turn_target_yaw:.2f}")
        self.state = "TURN_90_LEFT"
        self.stop()

    def start_scan_for_new_rows(self):
        self.get_logger().info("SCAN_FOR_NEW_ROWS")
        self.state = "SCAN_FOR_NEW_ROWS"
        self.scan_deadline = self.get_clock().now() + Duration(seconds=self.scan_timeout)
        self.new_left_wall_y = None
        self.new_right_wall_y = None
        self.stop()
    
    def align_for_row_entry(self):
        """Attend de voir une ligne fraîche pour reprendre le suivi."""
        if self.line_is_fresh() and len(self.line_points) >= 2:
            self.get_logger().info("Rang détecté après 2ᵉ virage → FOLLOW_LINE")
            self.state = "FOLLOW_LINE"
            return

        if self.wait_line_deadline and self.get_clock().now() >= self.wait_line_deadline:
            self.get_logger().warn("Pas de ligne après 2ᵉ virage → EXIT_FORWARD")
            self.start_exit_forward()
            return

        self.stop()
    

    def start_move_to_center(self):
        if self.new_left_wall_y is None or self.new_right_wall_y is None:
            self.get_logger().error("Pas de nouveaux murs -> STOPPED")
            self.state = "STOPPED"
            return
        
        center_y = (self.new_left_wall_y + self.new_right_wall_y) / 2.0
        self.target_y_offset = center_y
        
        self.get_logger().info(
            f"MOVE_TO_CENTER: L={self.new_left_wall_y:.2f} R={self.new_right_wall_y:.2f} C={center_y:.2f}"
        )
        self.state = "MOVE_TO_CENTER"

    # ================ BEHAVIORS ================

    def follow_line_step(self):
        self.get_logger().info(f"[DEBUG] follow_line_step | fresh={self.line_is_fresh()} | pts={len(self.line_points)}")

        if (not self.line_is_fresh()) or (len(self.line_points) < 2):
            self.get_logger().info(f"[DEBUG] Ligne trop courte : {len(self.line_points)} pts → WAIT_LINE")
            self.start_wait_line()
            return


        target = self.pick_target_point()
        if target is None:
            self.start_wait_line()
            return

        tx, ty = target
        angle = math.atan2(ty, tx)

        w = clamp(self.follow_k * angle, -self.max_w_follow, self.max_w_follow)
        w = self.apply_min_w(w, self.min_w)

        v = self.follow_v
        v_scale = max(0.15, 1.0 - abs(angle) / 0.9)
        v *= v_scale

        self.last_follow_w = float(w)
        self.publish_cmd(v, w)

    def wait_line_step(self):
        if self.line_is_fresh() and (len(self.line_points) >= 2) and (self.pick_target_point() is not None):
            self.get_logger().info("Ligne retrouvée -> FOLLOW_LINE")
            self.state = "FOLLOW_LINE"
            return

        if self.wait_line_deadline is not None and self.get_clock().now() >= self.wait_line_deadline:
            self.get_logger().warn("Ligne absente -> EXIT_FORWARD")
            self.start_exit_forward()
            return

        self.stop()

    def exit_forward_step(self):
        if self.exit_end_time is None:
            self.stop()
            self.state = "STOPPED"
            return

        if self.get_clock().now() >= self.exit_end_time:
            self.stop()

            if self.half_turns_done == 0:
                # Premier bout : demi-tour 180° pour faire le RETOUR dans le même rang
                self.half_turns_done = 1
                self.get_logger().info("BOUT 1 (ALLER) -> Demi-tour 180° pour RETOUR")
                self.ignore_markers = True  # Ignorer les markers pendant la rotation !
                self.turn_target_yaw = wrap_pi(self.yaw + math.pi)
                self.state = "TURN_90_RIGHT"  # On utilise le même état mais avec target = +180°
                self.stop()
            else:
                # Deuxième bout : RETOUR terminé
                self.rows_completed += 1
                self.get_logger().info(f"Rang {self.rows_completed} terminé (ALLER+RETOUR)")
                self.half_turns_done = 0  # Reset pour le prochain rang
                
                if self.serpentin_mode and self.rows_completed < self.max_rows:
                    # Le robot a terminé ALLER+RETOUR
                    # rotation_direction: -1=droite (Y+), +1=gauche (Y-)
                    target_turn = wrap_pi(self.yaw + self.rotation_direction * math.pi / 2.0)
                    
                    direction_name = "gauche (+90°)" if self.rotation_direction > 0 else "droite (-90°)"
                    self.get_logger().info(
                        f"Changement de rang: 1er virage {direction_name} vers {target_turn:.2f}"
                    )
                    
                    self.ignore_markers = True  # Ignorer pendant rotation
                    self.row_change_step = 2  # Premier 90°
                    self.turn_target_yaw = target_turn
                    self.state = "TURN_90_RIGHT"
                    self.stop()
                else:
                    self.get_logger().info("PARCOURS TERMINÉ")
                    self.state = "STOPPED"
            return

        # Calculer direction de sortie basée sur la ligne verte
        w = 0.0
        
        if self.line_is_fresh() and len(self.line_points) >= 2:
            # Calculer angle moyen de la ligne verte
            # On prend les points entre 1m et 3m devant
            valid_points = [(x, y) for (x, y) in self.line_points if 1.0 < x < 3.0]
            
            if len(valid_points) >= 2:
                # Régression linéaire simple : angle = atan2(Δy, Δx)
                sum_x = sum(p[0] for p in valid_points)
                sum_y = sum(p[1] for p in valid_points)
                sum_xx = sum(p[0]*p[0] for p in valid_points)
                sum_xy = sum(p[0]*p[1] for p in valid_points)
                n = len(valid_points)
                
                if n * sum_xx - sum_x * sum_x > 1e-6:
                    # Pente de la ligne : m = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
                    angle = math.atan(slope)  # Angle de la ligne par rapport à l'axe X
                    w = clamp(2.0 * angle, -self.exit_w_max, self.exit_w_max)
        
        # Fallback sur last_follow_w si pas de ligne
        if abs(w) < 0.01:
            w = clamp(self.last_follow_w, -self.exit_w_max, self.exit_w_max)
        
        self.publish_cmd(self.exit_v, w)

    def turn_90_right_step(self):
        if self.turn_target_yaw is None:
            self.state = "STOPPED"
            self.stop()
            return

        err = wrap_pi(self.turn_target_yaw - self.yaw)

        if abs(err) < self.turn_90_tol:
            self.stop()
            
            # Réactiver la détection des markers
            self.ignore_markers = False
            self.get_logger().info("Rotation terminée -> Réactivation markers")
            self.publish_state()
            self.get_logger().info("État publié après rotation.")


            # Cas 1: Demi-tour 180° après ALLER (pour faire RETOUR)
            if self.half_turns_done == 1 and self.row_change_step == 0:
                self.get_logger().info("180° terminé -> FOLLOW_LINE (RETOUR)")
                self.state = "FOLLOW_LINE"
                self.wait_line_deadline = self.get_clock().now() + Duration(seconds=0.5)
                self.get_logger().info(f"Nouvel état = {self.state}")
                self.publish_state() # petit délai
                self.get_logger().info(f"[DEBUG] 180° → ignore_markers = {self.ignore_markers}")
                self.get_logger().info(f"[DEBUG] line_points = {len(self.line_points)} pts | line_is_fresh = {self.line_is_fresh()}")

                return
            
            
            # Cas 2: Premier -90° pour changement de rang → avancer latéralement
            elif self.row_change_step == 2:
                self.get_logger().info(f"1er -90° OK -> MOVE_LATERAL ({self.inter_row_distance}m)")
                self.row_change_step = 3
                # Capturer position de départ
                self.lateral_start_x = self.x
                self.lateral_start_y = self.y
                self.lateral_target_distance = self.inter_row_distance
                self.state = "MOVE_LATERAL"
                
            
            # Cas 3: Deuxième -90° après avance latérale
            elif self.row_change_step == 4:
                self.get_logger().info("2ème -90° OK -> ALIGN_FOR_ROW_ENTRY")
                self.row_change_step = 0
                self.ignore_markers = False

                self.line_points = []
                self.last_line_time = None
                
                # Ajoute un nouvel état qui attend une ligne fraîche pour bien se réaligner
                self.state = "ALIGN_FOR_ROW_ENTRY"
                self.wait_line_deadline = self.get_clock().now() + Duration(seconds=2.0)
                return
      
              
            # Fallback
            else:
                self.ignore_markers = False
                self.get_logger().info("Rotation terminée -> FOLLOW_LINE")
                self.state = "FOLLOW_LINE"
                self.wait_line_deadline = self.get_clock().now() + Duration(seconds=0.5)
            return

        w = clamp(self.turn_90_k * err, -self.turn_90_max_w, self.turn_90_max_w)
        w = self.apply_min_w(w, self.turn_90_min_w)
        self.publish_cmd(0.0, w)

    def turn_90_left_step(self):
        if self.turn_target_yaw is None:
            self.state = "STOPPED"
            self.stop()
            return

        err = wrap_pi(self.turn_target_yaw - self.yaw)

        if abs(err) < self.turn_90_tol:
            self.stop()
            self.get_logger().info("90° LEFT OK -> FOLLOW_LINE")
            self.state = "FOLLOW_LINE"
            return

        w = clamp(self.turn_90_k * err, -self.turn_90_max_w, self.turn_90_max_w)
        w = self.apply_min_w(w, self.turn_90_min_w)
        self.publish_cmd(0.0, w)

    def scan_for_new_rows_step(self):
        """
        Cherche la ligne centrale du nouveau rang.
        Au lieu de détecter les murs, on utilise directement center_path !
        """
        # Timeout ?
        if self.scan_deadline and self.get_clock().now() >= self.scan_deadline:
            self.get_logger().error("SCAN timeout -> STOPPED")
            self.state = "STOPPED"
            self.stop()
            return

        # La ligne centrale est-elle disponible ?
        if not self.line_is_fresh() or len(self.line_points) < 2:
            self.stop()
            return

        # Chercher le point de la ligne centrale devant nous (x > 0)
        # On veut le point le plus proche sur X (pour avancer perpendiculairement)
        best_point = None
        best_x = 1e9
        
        for (x, y) in self.line_points:
            if x > 0.3:  # Devant le robot, pas trop proche
                if x < best_x:
                    best_x = x
                    best_point = (x, y)
        
        if best_point is None:
            self.stop()
            return
        
        # On a trouvé le point cible !
        target_x, target_y = best_point
        self.get_logger().info(
            f"Ligne centrale détectée ! Point cible: x={target_x:.2f} y={target_y:.2f}"
        )
        
        # Passer en mode avance vers ce point
        self.row_change_step = 3
        self.state = "MOVE_TO_CENTERLINE"

    def move_lateral_step(self):
        """
        Avance latéralement (perpendiculaire aux rangs) sur inter_row_distance.
        """
        if self.lateral_start_x is None or self.lateral_start_y is None:
            self.stop()
            self.state = "STOPPED"
            return
        
        # Distance parcourue depuis le départ
        dx = self.x - self.lateral_start_x
        dy = self.y - self.lateral_start_y
        dist_traveled = math.hypot(dx, dy)
        
        if dist_traveled >= self.lateral_target_distance:
            self.stop()
            self.get_logger().info(f"Distance latérale parcourue: {dist_traveled:.2f}m -> 2ème virage")
            # Deuxième rotation dans le même sens
            target_final = wrap_pi(self.yaw + self.rotation_direction * math.pi / 2.0)
            direction_name = "gauche (+90°)" if self.rotation_direction > 0 else "droite (-90°)"
            self.get_logger().info(f"2ème virage {direction_name}: de {self.yaw:.2f} vers {target_final:.2f}")
            self.row_change_step = 4
            self.turn_target_yaw = target_final
            self.state = "TURN_90_RIGHT"
            return
        
        # Continuer d'avancer tout droit
        v = self.center_v
        self.publish_cmd(v, 0.0)

    def align_for_turn_step(self):
        """Réaligne le robot avec la direction de sortie du rang avant de tourner."""
        if self.turn_target_yaw is None:
            self.state = "STOPPED"
            self.stop()
            return

        err = wrap_pi(self.turn_target_yaw - self.yaw)

        if abs(err) < self.turn_90_tol:
            self.stop()
            self.get_logger().info("Alignement OK -> TURN_90_RIGHT (1/2)")
            self.row_change_step = 2  # Premier 90°
            self.start_turn_90_right()
            return

        w = clamp(self.turn_90_k * err, -self.turn_90_max_w, self.turn_90_max_w)
        w = self.apply_min_w(w, self.turn_90_min_w)
        self.publish_cmd(0.0, w)

    # ================ MAIN LOOP ================

    def loop(self):
        if not self.have_odom:
            return
        if self.state == "ALIGN_FOR_ROW_ENTRY":
            self.align_for_row_entry()
            return


        if self.state in ["TURN_TO_GOAL", "DRIVE_STRAIGHT", "FINAL_YAW"]:
            self.go_to_pose_logic()
            return

        if self.state == "FOLLOW_LINE":
            self.follow_line_step()
            return

        if self.state == "WAIT_LINE":
            self.wait_line_step()
            return

        if self.state == "EXIT_FORWARD":
            self.exit_forward_step()
            return

        if self.state == "TURN_90_RIGHT":
            self.turn_90_right_step()
            return

        if self.state == "TURN_90_LEFT":
            self.turn_90_left_step()
            return

        if self.state == "MOVE_LATERAL":
            self.move_lateral_step()
            return

        if self.state == "SCAN_FOR_NEW_ROWS":
            # Obsolète mais on le garde
            self.get_logger().warn("État SCAN obsolète")
            self.state = "STOPPED"
            return

        if self.state == "MOVE_TO_CENTERLINE":
            # Obsolète
            self.get_logger().warn("État MOVE_TO_CENTERLINE obsolète")
            self.state = "STOPPED"
            return

        if self.state == "MOVE_TO_CENTER":
            # Obsolète
            self.get_logger().warn("État MOVE_TO_CENTER obsolète")
            self.state = "STOPPED"
            return

        if self.state == "ALIGN_FOR_TURN":
            self.align_for_turn_step()
            return

        if self.state == "STOPPED":
            self.stop()
            return

    def go_to_pose_logic(self):
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        dist = math.hypot(dx, dy)

        yaw_to_goal = math.atan2(dy, dx)
        yaw_err_to_goal = wrap_pi(yaw_to_goal - self.yaw)
        yaw_err_final = wrap_pi(self.goal_yaw - self.yaw)

        if self.state == "TURN_TO_GOAL":
            if dist < self.pos_tol:
                self.state = "FINAL_YAW"
                self.stop()
                return

            if abs(yaw_err_to_goal) < self.yaw_tol:
                self.state = "DRIVE_STRAIGHT"
                self.stop()
                return

            w = clamp(self.k_yaw * yaw_err_to_goal, -self.max_w, self.max_w)
            w = self.apply_min_w(w, self.min_w)
            self.publish_cmd(0.0, w)
            return

        if self.state == "DRIVE_STRAIGHT":
            if dist < self.pos_tol:
                self.state = "FINAL_YAW"
                self.stop()
                return

            if abs(yaw_err_to_goal) > self.yaw_back_to_turn:
                self.state = "TURN_TO_GOAL"
                self.stop()
                return

            v = clamp(self.k_lin * dist, 0.0, self.max_v)
            v_scale = max(0.2, 1.0 - abs(yaw_err_to_goal) / 0.8)
            v *= v_scale

            w = clamp(self.k_yaw_drive * yaw_err_to_goal, -0.8, 0.8)
            self.publish_cmd(v, w)
            return

        if self.state == "FINAL_YAW":
            if abs(yaw_err_final) < self.final_yaw_tol:
                self.stop()
                self.get_logger().info("Goal atteint -> FOLLOW_LINE")
                self.state = "FOLLOW_LINE"
                return

            w = clamp(self.k_yaw_final * yaw_err_final, -self.max_w_final, self.max_w_final)
            w = self.apply_min_w(w, self.min_w)
            self.publish_cmd(0.0, w)
            return


def main():
    rclpy.init()
    node = GoToPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()