#!/usr/bin/env python3
"""
markers.py - Création de markers RViz pour visualisation
=========================================================

Génère des markers pour:
- Murs gauche et droite (LINE_STRIP)
- Chemin central (LINE_STRIP)
- Informations textuelles (TEXT_VIEW_FACING)
"""

import rclpy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from .ransac import y_on_line


class MarkerPublisher:
    """
    Gestionnaire de publication de markers pour RViz.
    """
    
    def __init__(self, node, topic='/scout/debug_markers'):
        """
        Initialise le publisher de markers.
        
        Args:
            node: Node ROS2
            topic: Topic de publication
        """
        self.node = node
        self.pub_markers = node.create_publisher(MarkerArray, topic, 10)
        self.marker_stamp_latest = True
        self.roi_x_min = 0.25
        self.roi_x_max = 12.0
        
    def set_params(self, params_dict):
        """
        Configure les paramètres du publisher.
        
        Args:
            params_dict: Dictionnaire des paramètres
        """
        self.marker_stamp_latest = params_dict.get('marker_stamp_latest', True)
        self.roi_x_min = params_dict.get('roi_x_min', 0.25)
        self.roi_x_max = params_dict.get('roi_x_max', 12.0)
        
    def _marker_header(self, header):
        """
        Crée un header pour les markers.
        
        Args:
            header: Header du message source
            
        Returns:
            Header modifié ou original
        """
        if not self.marker_stamp_latest:
            return header
        hdr = Header()
        hdr.frame_id = header.frame_id
        hdr.stamp = rclpy.time.Time().to_msg()  # stamp=0 => TF latest
        return hdr
        
    def publish_markers(self, header, left_line, right_line, centers,
                       inl_left, inl_right, obstacle_side=0, pts_count=0):
        """
        Publie les markers de visualisation.
        
        Args:
            header: Header du message source
            left_line, right_line: Tuples (a, b, c) des lignes ou None
            centers: Liste de tuples (x, y) du chemin central
            inl_left, inl_right: Nombre d'inliers pour chaque mur
            obstacle_side: -1, 0, ou +1 pour direction d'évitement
            pts_count: Nombre total de points traités
        """
        markers = MarkerArray()
        hdr = self._marker_header(header)

        # 1. Texte d'information
        m0 = self._create_info_text(hdr, pts_count, inl_left, inl_right, 
                                    len(centers), obstacle_side)
        markers.markers.append(m0)

        # 2. Mur gauche
        if left_line is not None:
            mL = self._create_wall_marker(hdr, left_line, 'wall_left', 1, 
                                          color=(1.0, 0.0, 1.0))
            markers.markers.append(mL)

        # 3. Mur droit
        if right_line is not None:
            mR = self._create_wall_marker(hdr, right_line, 'wall_right', 2,
                                          color=(0.0, 0.0, 1.0))
            markers.markers.append(mR)

        # 4. Chemin central
        mC = self._create_center_path_marker(hdr, centers)
        markers.markers.append(mC)

        self.pub_markers.publish(markers)
        
    def _create_info_text(self, header, pts_count, inl_left, inl_right, 
                         centers_count, obstacle_side):
        """
        Crée le marker texte avec les informations de debug.
        
        Args:
            header: Header du marker
            pts_count: Nombre de points total
            inl_left, inl_right: Nombres d'inliers
            centers_count: Nombre de points dans le chemin
            obstacle_side: Direction d'évitement
            
        Returns:
            Marker TEXT_VIEW_FACING
        """
        m = Marker()
        m.header = header
        m.ns = "info"
        m.id = 0
        m.type = Marker.TEXT_VIEW_FACING
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 1.6
        m.scale.z = 0.25
        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0

        side_txt = "none"
        if obstacle_side == -1:
            side_txt = "bias RIGHT (obs left)"
        elif obstacle_side == +1:
            side_txt = "bias LEFT (obs right)"

        m.text = (f"pts={pts_count} | inL={inl_left} inR={inl_right} | "
                 f"centers={centers_count} | {side_txt}")
        return m
        
    def _create_wall_marker(self, header, line, namespace, marker_id, color):
        """
        Crée un marker LINE_STRIP pour un mur.
        
        Args:
            header: Header du marker
            line: Tuple (a, b, c) de la ligne
            namespace: Namespace du marker
            marker_id: ID unique du marker
            color: Tuple (r, g, b)
            
        Returns:
            Marker LINE_STRIP
        """
        m = Marker()
        m.header = header
        m.ns = namespace
        m.id = marker_id
        m.type = Marker.LINE_STRIP
        m.scale.x = 0.08
        m.color.a = 1.0
        m.color.r = color[0]
        m.color.g = color[1]
        m.color.b = color[2]

        # Échantillonner la ligne sur plusieurs x
        xs = [self.roi_x_min, 
              (self.roi_x_min + self.roi_x_max) * 0.5, 
              self.roi_x_max]
              
        for x in xs:
            y = y_on_line(line, x)
            if y is None:
                continue
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.15
            m.points.append(p)
            
        return m
        
    def _create_center_path_marker(self, header, centers):
        """
        Crée un marker LINE_STRIP pour le chemin central.
        
        Args:
            header: Header du marker
            centers: Liste de tuples (x, y)
            
        Returns:
            Marker LINE_STRIP
        """
        m = Marker()
        m.header = header
        m.ns = "center_path"
        m.id = 3
        m.type = Marker.LINE_STRIP
        m.scale.x = 0.08
        m.color.a = 1.0
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0

        # Point de départ à l'origine
        p0 = Point()
        p0.x = 0.0
        p0.y = 0.0
        p0.z = 0.12
        m.points.append(p0)

        # Ajout des points du chemin
        for x, y in centers:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.12
            m.points.append(p)
            
        return m