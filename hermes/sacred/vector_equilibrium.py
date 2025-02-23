"""
Vector Equilibrium (Cuboctahedron) implementation for Hermes AI system.
Represents the fundamental equilibrium of energy dynamics.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from hermes.utils.logging import logger

import numpy as np


@dataclass
class VectorForce:
    """Represents a force vector in the equilibrium."""

    direction: np.ndarray
    magnitude: float
    frequency: float
    phase: float


class VectorEquilibrium:
    """
    Implementation of Vector Equilibrium (Cuboctahedron) sacred geometry.
    Represents perfect equilibrium of forces and energies.
    """

    def __init__(self, size: float = 1.0):
        self.size = size
        self.center = np.array([0.0, 0.0, 0.0])
        self.vertices = self._generate_vertices()
        self.edges = self._generate_edges()
        self.faces = self._generate_faces()
        self.forces = self._generate_forces()
        self.frequency = 528.0  # Hz - DNA repair frequency

    def _generate_vertices(self) -> np.ndarray:
        """Generate the 12 vertices of the Vector Equilibrium."""
        # Create the 12 vertices in equilibrium
        vertices = []

        # Square in middle xy-plane
        square1 = [[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]]

        # Top and bottom squares, rotated 45 degrees
        square2 = [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1]]

        square3 = [[1, 0, -1], [0, 1, -1], [-1, 0, -1], [0, -1, -1]]

        vertices.extend(square1)
        vertices.extend(square2)
        vertices.extend(square3)

        return np.array(vertices) * self.size

    def _generate_edges(self) -> List[Tuple[int, int]]:
        """Generate the 24 edges connecting vertices."""
        edges = []

        # Connect vertices in each square
        for i in range(0, 12, 4):
            for j in range(4):
                edges.append((i + j, i + (j + 1) % 4))

        # Connect squares
        for i in range(4):
            edges.append((i, i + 4))
            edges.append((i, i + 8))
            edges.append((i + 4, i + 8))

        return edges

    def _generate_faces(self) -> List[List[int]]:
        """Generate the 14 faces (8 triangular, 6 square)."""
        faces = []

        # Square faces
        square_faces = [
            [0, 1, 2, 3],  # Middle
            [4, 5, 6, 7],  # Top
            [8, 9, 10, 11],  # Bottom
            [0, 4, 1, 5],  # Front
            [2, 6, 3, 7],  # Back
            [1, 5, 2, 6],  # Right
        ]
        faces.extend(square_faces)

        # Triangular faces
        for i in range(4):
            # Top triangles
            faces.append([i, (i + 1) % 4, i + 4])
            # Bottom triangles
            faces.append([i, (i + 1) % 4, i + 8])

        return faces

    def _generate_forces(self) -> List[VectorForce]:
        """Generate the 12 equilibrium force vectors."""
        forces = []

        # Create force vectors from center to each vertex
        for vertex in self.vertices:
            direction = vertex / np.linalg.norm(vertex)
            force = VectorForce(
                direction=direction, magnitude=1.0, frequency=self.frequency, phase=0.0
            )
            forces.append(force)

        return forces

    def calculate_energy_field(self, point: np.ndarray) -> float:
        """Calculate energy field strength at given point."""
        total_energy = 0.0

        # Contribution from vertices
        for vertex in self.vertices:
            dist = np.linalg.norm(point - vertex)
            total_energy += self.frequency / (1 + dist)

        # Contribution from edges
        for edge in self.edges:
            v1 = self.vertices[edge[0]]
            v2 = self.vertices[edge[1]]
            dist = self._point_to_line_distance(point, v1, v2)
            total_energy += self.frequency / (1 + dist)

        return total_energy

    def _point_to_line_distance(
        self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray
    ) -> float:
        """Calculate distance from point to line segment."""
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_length = np.linalg.norm(line_vec)
        line_unit_vec = line_vec / line_length
        projection = np.dot(point_vec, line_unit_vec)

        if projection <= 0:
            return np.linalg.norm(point_vec)
        elif projection >= line_length:
            return np.linalg.norm(point - line_end)
        else:
            return np.linalg.norm(point_vec - projection * line_unit_vec)

    def get_force_field(self, point: np.ndarray) -> np.ndarray:
        """Calculate net force vector at given point."""
        net_force = np.zeros(3)

        for force in self.forces:
            # Calculate distance-based attenuation
            dist = np.linalg.norm(point - self.center)
            attenuation = 1 / (1 + dist**2)

            # Add force contribution
            net_force += force.direction * force.magnitude * attenuation

        return net_force

    def get_resonant_frequencies(self) -> List[float]:
        """Calculate resonant frequencies of the structure."""
        base_freq = self.frequency
        return [base_freq * (n + 1) for n in range(12)]

    def get_equilibrium_points(self) -> List[Dict[str, Any]]:
        """Get key equilibrium points with their energies."""
        points = [
            {
                "position": self.center,
                "energy": self.frequency,
                "description": "Central equilibrium point",
            }
        ]

        # Add vertex points
        for i, vertex in enumerate(self.vertices):
            points.append(
                {
                    "position": vertex,
                    "energy": self.frequency,
                    "description": f"Vertex {i+1}",
                }
            )

        return points

    def apply_quantum_rotation(self, angle: float, axis: str) -> None:
        """Apply quantum rotation to the equilibrium structure."""
        # Create rotation matrix
        c = np.cos(angle)
        s = np.sin(angle)

        if axis == "x":
            rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == "y":
            rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else:  # z
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # Apply rotation to vertices and force directions
        self.vertices = np.dot(self.vertices, rotation.T)
        for force in self.forces:
            force.direction = np.dot(force.direction, rotation.T)

    def calculate_torsion_field(self, points: np.ndarray) -> np.ndarray:
        """Calculate torsion field strength across multiple points."""
        field = np.zeros(points.shape[:-1])

        for idx in np.ndindex(points.shape[:-1]):
            point = points[idx]
            # Calculate curl of force field
            h = 0.001  # Small displacement for numerical derivative

            # Calculate curl components
            curl_x = (
                self.get_force_field(point + np.array([0, h, 0]))[2]
                - self.get_force_field(point - np.array([0, h, 0]))[2]
            ) / (2 * h)
            curl_y = (
                self.get_force_field(point + np.array([0, 0, h]))[0]
                - self.get_force_field(point - np.array([0, 0, h]))[0]
            ) / (2 * h)
            curl_z = (
                self.get_force_field(point + np.array([h, 0, 0]))[1]
                - self.get_force_field(point - np.array([h, 0, 0]))[1]
            ) / (2 * h)

            field[idx] = np.sqrt(curl_x**2 + curl_y**2 + curl_z**2)

        return field
