#!/usr/bin/env python3
"""
planner.py - Planification de trajectoire et évitement d'obstacles
===================================================================

Gère:
- Stabilisation gauche/droite des murs détectés (anti-flip)
- Génération du chemin central (centerline)
- Détection et évitement d'obstacles
- Lissage temporel et hold des trajectoires
"""

import math
from .ransac import clamp, y_on_line


class PathPlanner:
    """
    Planificateur de trajectoire pour navigation en couloir.
    
    Maintient un état interne pour la stabilisation temporelle.
    """
    
    def __init__(self, node):
        """
        Initialise le planificateur.
        
        Args:
            node: Node ROS2 pour accès aux paramètres et clock
        """
        self.node = node
        
        # Paramètres de stabilisation des offsets
        self.x_ref_swap = None
        self.swap_hysteresis = None
        self.offset_lpf = None
        self.offset_max_step = None
        
        # Paramètres centerline
        self.center_x_start = None
        self.center_x_step = None
        self.center_x_count = None
        self.center_lpf = None
        self.center_hold_time = None
        
        # Paramètres robot
        self.robot_width = None
        self.safety_margin = None
        
        # Paramètres évitement
        self.avoid_enable = None
        self.avoid_bias = None
        self.avoid_apply_first_n = None
        self.obs_hyst = None
        self.obs_hold_time = None
        
        # ROI
        self.roi_x_min = None
        self.roi_x_max = None
        
        # État mémoire
        self.left_offset_mem = None
        self.right_offset_mem = None
        self.centers_mem = None
        self.last_centers = None
        self.last_centers_t = None
        self.obs_side_mem = 0
        self.obs_candidate = 0
        self.obs_candidate_t = 0.0
        
    def load_params(self, params_dict):
        """
        Charge les paramètres depuis un dictionnaire.
        
        Args:
            params_dict: Dictionnaire des paramètres
        """
        self.x_ref_swap = params_dict.get('x_ref_swap', 1.2)
        self.swap_hysteresis = params_dict.get('swap_hysteresis', 0.05)
        self.offset_lpf = params_dict.get('offset_lpf', 0.25)
        self.offset_max_step = params_dict.get('offset_max_step', 0.10)
        
        self.center_x_start = params_dict.get('center_x_start', 0.6)
        self.center_x_step = params_dict.get('center_x_step', 0.5)
        self.center_x_count = max(2, params_dict.get('center_x_count', 12))
        self.center_lpf = params_dict.get('center_lpf', 0.25)
        self.center_hold_time = params_dict.get('center_hold_time', 0.6)
        
        self.robot_width = params_dict.get('robot_width', 0.70)
        self.safety_margin = params_dict.get('safety_margin', 0.35)
        
        self.avoid_enable = params_dict.get('avoid_enable', True)
        self.avoid_bias = params_dict.get('avoid_bias', 0.85)
        self.avoid_apply_first_n = params_dict.get('avoid_apply_first_n', 6)
        self.obs_hyst = params_dict.get('obs_hyst', 0.12)
        self.obs_hold_time = params_dict.get('obs_hold_time', 0.8)
        
        self.roi_x_min = params_dict.get('roi_x_min', 0.25)
        self.roi_x_max = params_dict.get('roi_x_max', 12.0)
        
    def _now_sec(self):
        """Retourne le temps actuel en secondes."""
        return self.node.get_clock().now().nanoseconds * 1e-9
        
    def stabilize_walls(self, left_line, right_line, left_in, right_in):
        """
        Stabilise l'assignation gauche/droite des murs.
        
        Applique:
        - Anti-flip basé sur position relative
        - Mémoire pour éviter les swaps erratiques
        - Rate limiting sur les offsets
        
        Args:
            left_line, right_line: Tuples (a, b, c) des lignes
            left_in, right_in: Listes d'inliers
            
        Returns:
            Tuple (left_line, right_line, left_in, right_in) stabilisés
        """
        xref = max(self.roi_x_min + 0.2, min(self.roi_x_max - 0.2, self.x_ref_swap))
        yl_raw = y_on_line(left_line, xref)
        yr_raw = y_on_line(right_line, xref)

        if yl_raw is None or yr_raw is None:
            return left_line, right_line, left_in, right_in

        yl = yl_raw
        yr = yr_raw

        # Test 1: inversion directe (gauche < droite)
        if (yl + self.swap_hysteresis) < (yr - self.swap_hysteresis):
            left_line, right_line = right_line, left_line
            left_in, right_in = right_in, left_in
            yl, yr = yr, yl

        # Test 2: cohérence avec mémoire
        if self.left_offset_mem is not None and self.right_offset_mem is not None:
            d_keep = abs(yl - self.left_offset_mem) + abs(yr - self.right_offset_mem)
            d_swap = abs(yl - self.right_offset_mem) + abs(yr - self.left_offset_mem)
            if d_swap + 0.10 < d_keep:
                left_line, right_line = right_line, left_line
                left_in, right_in = right_in, left_in
                yl, yr = yr, yl

        # Rate limiting
        ms = max(0.01, self.offset_max_step)
        if self.left_offset_mem is not None:
            yl = clamp(yl, self.left_offset_mem - ms, self.left_offset_mem + ms)
        if self.right_offset_mem is not None:
            yr = clamp(yr, self.right_offset_mem - ms, self.right_offset_mem + ms)

        # Mise à jour mémoire avec LPF
        self._update_offsets_memory(yl, yr)

        return left_line, right_line, left_in, right_in

    def _update_offsets_memory(self, yl, yr):
        """Mise à jour filtrée des offsets mémorisés."""
        alpha = clamp(self.offset_lpf, 0.0, 1.0)
        if self.left_offset_mem is None or self.right_offset_mem is None:
            self.left_offset_mem = yl
            self.right_offset_mem = yr
            return
        self.left_offset_mem = (1.0 - alpha) * self.left_offset_mem + alpha * yl
        self.right_offset_mem = (1.0 - alpha) * self.right_offset_mem + alpha * yr

    def compute_centerline(self, left_line, right_line):
        """
        Calcule les points de la ligne centrale entre deux murs.
        
        Args:
            left_line, right_line: Tuples (a, b, c) des lignes
            
        Returns:
            Liste de tuples (x, y) formant le chemin central
        """
        centers = []
        min_dist_to_wall = (self.robot_width / 2.0) + self.safety_margin

        xs = [self.center_x_start + i * self.center_x_step 
              for i in range(self.center_x_count)]
              
        for x in xs:
            if x < self.roi_x_min or x > self.roi_x_max:
                continue
            yl = y_on_line(left_line, x)
            yr = y_on_line(right_line, x)
            if yl is None or yr is None:
                continue

            cy = 0.5 * (yl + yr)

            # Application de la marge de sécurité
            if abs(yl - cy) < min_dist_to_wall:
                cy = yl - min_dist_to_wall
            if abs(yr - cy) < min_dist_to_wall:
                cy = yr + min_dist_to_wall

            centers.append((x, cy))

        return centers

    def apply_obstacle_avoidance(self, centers, obstacle_points, wall_keys, xref):
        """
        Applique un biais d'évitement d'obstacles sur la trajectoire.
        
        Args:
            centers: Liste de tuples (x, y) du chemin central
            obstacle_points: Liste de tuples (x, y) des obstacles détectés
            wall_keys: Set des clés spatiales des murs (pour exclusion)
            xref: Position x de référence
            
        Returns:
            Tuple (centers_modifiés, obstacle_side) où:
            - centers_modifiés: Nouvelle liste de points
            - obstacle_side: -1 (obs gauche), 0 (aucun), +1 (obs droite)
        """
        if not self.avoid_enable or not centers:
            return centers, self.obs_side_mem

        if not obstacle_points:
            return centers, self.obs_side_mem

        # Calcul ymean des obstacles
        ymean = sum(p[1] for p in obstacle_points) / len(obstacle_points)
        
        # Décision stable du côté
        obstacle_side = self._stable_obstacle_side(ymean)

        # Calcul de l'urgence
        cy_ref = 0.0
        if centers:
            best = min(centers, key=lambda p: abs(p[0] - xref))
            cy_ref = best[1]

        dmin = min(abs(p[1] - cy_ref) for p in obstacle_points)
        urg = clamp((0.8 - dmin) / 0.8, 0.0, 1.0)
        bias = self.avoid_bias * (0.3 + 0.7 * urg) * float(obstacle_side)

        # Application en rampe sur les N premiers points
        n_apply = max(1, self.avoid_apply_first_n)
        new_centers = []
        for i, (x, y) in enumerate(centers):
            if i < n_apply:
                s = float(i) / float(max(1, n_apply - 1))
            else:
                s = 1.0
            new_centers.append((x, y + s * bias))

        return new_centers, obstacle_side

    def _stable_obstacle_side(self, ymean):
        """
        Détermine le côté de l'obstacle de manière stable.
        
        ymean > +hyst => obstacle à gauche => bias RIGHT (-1)
        ymean < -hyst => obstacle à droite => bias LEFT (+1)
        
        Args:
            ymean: Position moyenne Y des obstacles
            
        Returns:
            -1, 0, ou +1
        """
        t = self._now_sec()

        if ymean > self.obs_hyst:
            candidate = -1
        elif ymean < -self.obs_hyst:
            candidate = +1
        else:
            candidate = 0

        # Reset timer si changement
        if candidate != self.obs_candidate:
            self.obs_candidate = candidate
            self.obs_candidate_t = t

        # Accepter si stable assez longtemps
        if (t - self.obs_candidate_t) >= self.obs_hold_time:
            self.obs_side_mem = self.obs_candidate

        return self.obs_side_mem

    def smooth_centers(self, centers):
        """
        Applique un filtre passe-bas sur les points du chemin.
        
        Args:
            centers: Liste de tuples (x, y)
            
        Returns:
            Liste de tuples (x, y) lissés
        """
        if centers is None or len(centers) < 2:
            return centers if centers else []

        alpha = clamp(self.center_lpf, 0.0, 1.0)
        if self.centers_mem is None or len(self.centers_mem) != len(centers):
            self.centers_mem = centers
            return centers

        smooth = []
        for (x, y), (xm, ym) in zip(centers, self.centers_mem):
            y2 = (1.0 - alpha) * ym + alpha * y
            smooth.append((x, y2))
        self.centers_mem = smooth
        return smooth

    def hold_centers_if_needed(self, centers):
        """
        Maintient le dernier chemin valide pendant un certain temps.
        
        Args:
            centers: Liste de tuples (x, y) ou None
            
        Returns:
            Liste de tuples (x, y) (courante ou maintenue)
        """
        t = self._now_sec()
        if centers is None:
            centers = []

        if len(centers) >= 2:
            self.last_centers = centers
            self.last_centers_t = t
            return centers

        if self.last_centers is not None and self.last_centers_t is not None:
            if (t - self.last_centers_t) <= self.center_hold_time:
                return self.last_centers
        return centers