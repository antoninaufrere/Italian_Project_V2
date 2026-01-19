#!/usr/bin/env python3
"""
ransac.py - Détection de lignes (murs) par RANSAC 2D
=====================================================

Implémente RANSAC pour détecter des lignes dans des nuages de points 2D (XY).
Utilisé pour détecter les murs/rangées dans un couloir.
"""

import math
import random


def clamp(v, vmin, vmax):
    """Limite une valeur entre min et max."""
    return max(vmin, min(vmax, v))


def wrap_pi(a):
    """Normalise un angle dans [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def fit_line_from_2pts(p1, p2):
    """
    Calcule une ligne 2D à partir de 2 points.
    
    Forme: ax + by + c = 0, avec sqrt(a^2 + b^2) = 1
    
    Args:
        p1, p2: Tuples (x, y)
        
    Returns:
        Tuple (a, b, c) ou None si les points sont identiques
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    # normal (a,b) = (dy, -dx)
    a = dy
    b = -dx
    n = math.hypot(a, b)
    if n < 1e-9:
        return None
    a /= n
    b /= n
    c = -(a * x1 + b * y1)
    return (a, b, c)


def point_line_dist(line, p):
    """
    Calcule la distance point-ligne.
    
    Args:
        line: Tuple (a, b, c)
        p: Tuple (x, y)
        
    Returns:
        Distance perpendiculaire
    """
    a, b, c = line
    x, y = p
    return abs(a * x + b * y + c)


def line_angle(line):
    """
    Calcule l'angle d'orientation d'une ligne.
    
    Args:
        line: Tuple (a, b, c)
        
    Returns:
        Angle en radians [-pi, pi]
    """
    # direction = (-b, a)
    a, b, _ = line
    return math.atan2(a, -b)


def y_on_line(line, x):
    """
    Calcule y pour un x donné sur la ligne.
    
    Args:
        line: Tuple (a, b, c)
        x: Coordonnée x
        
    Returns:
        Coordonnée y correspondante ou None si ligne verticale
    """
    a, b, c = line
    if abs(b) < 1e-9:
        return None
    return -(a * x + c) / b


def ransac_line(points, iters=260, dist_th=0.08, min_inliers=80,
                theta_hint=None, theta_max_dev_deg=12.0):
    """
    Détecte une ligne dans un nuage de points 2D par RANSAC.
    
    Args:
        points: Liste de tuples (x, y)
        iters: Nombre d'itérations RANSAC
        dist_th: Seuil de distance pour être inlier (mètres)
        min_inliers: Nombre minimum d'inliers pour accepter une ligne
        theta_hint: Angle de référence pour filtrage (radians)
        theta_max_dev_deg: Déviation max par rapport à theta_hint (degrés)
        
    Returns:
        Tuple (line, inliers) où:
        - line: Tuple (a, b, c) ou None
        - inliers: Liste de points inliers
    """
    if len(points) < 2:
        return None, []

    best_line = None
    best_in = []
    n = len(points)
    theta_max_dev = math.radians(max(1.0, theta_max_dev_deg))

    for _ in range(iters):
        # Sélection aléatoire de 2 points
        i1 = random.randrange(n)
        i2 = random.randrange(n)
        if i1 == i2:
            continue
            
        line = fit_line_from_2pts(points[i1], points[i2])
        if line is None:
            continue

        # Vérification de l'angle si hint fourni
        if theta_hint is not None:
            th = line_angle(line)
            if abs(wrap_pi(th - theta_hint)) > theta_max_dev:
                continue

        # Comptage des inliers
        inliers = []
        for p in points:
            if point_line_dist(line, p) <= dist_th:
                inliers.append(p)

        # Mise à jour du meilleur modèle
        if len(inliers) > len(best_in):
            best_in = inliers
            best_line = line

    if best_line is None or len(best_in) < min_inliers:
        return None, []
    return best_line, best_in


def remove_inliers(points, inliers, eps=1e-4):
    """
    Retire les inliers d'une liste de points.
    
    Utilise une grille spatiale pour la comparaison efficace.
    
    Args:
        points: Liste de tuples (x, y)
        inliers: Liste de tuples (x, y) à retirer
        eps: Epsilon pour la grille spatiale
        
    Returns:
        Liste de points restants
    """
    def key(p):
        return (int(p[0] / eps), int(p[1] / eps))
    s = set(key(p) for p in inliers)
    return [p for p in points if key(p) not in s]