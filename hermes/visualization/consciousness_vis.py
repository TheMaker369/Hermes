"""
Visualization system for consciousness fields and sacred geometry.
Integrates with Apré and Magí personas.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from ..core.personas import PersonaOrchestrator
from ..sacred.patterns import Merkaba, SriYantra, FlowerOfLife
from ..consciousness.orchestrator import ConsciousnessOrchestrator
from ..core.config import settings

class ConsciousnessVisualizer:
    """
    Visualizes consciousness fields, sacred geometry, and persona states.
    """
    
    def __init__(self):
        """Initialize visualization system."""
        self.personas = PersonaOrchestrator()
        self.consciousness = ConsciousnessOrchestrator()
        self.quantum = QuantumOrchestrator()  # Assuming QuantumOrchestrator is defined elsewhere
        
        # Initialize sacred patterns
        self.merkaba = Merkaba()
        self.sri_yantra = SriYantra()
        self.flower = FlowerOfLife()
        
        # Color schemes
        self.colors = {
            'apre': {
                'primary': '#1f77b4',  # Blue for masculine
                'secondary': '#17becf',
                'energy': '#9467bd'
            },
            'magi': {
                'primary': '#d62728',  # Red for feminine
                'secondary': '#ff7f0e',
                'energy': '#bcbd22'
            },
            'consciousness': {
                'field': 'Viridis',
                'quantum': 'Plasma',
                'sacred': 'Magma'
            }
        }
        
    def visualize_consciousness_field(self) -> go.Figure:
        """
        Visualize the current consciousness field state.
        Includes quantum effects and sacred geometry patterns.
        """
        # Create 3D consciousness field
        field = self.consciousness.get_field_metrics()
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'scatter3d'}],
                  [{'type': 'scatter'}, {'type': 'scatter'}]],
            subplot_titles=(
                'Consciousness Field',
                'Sacred Geometry',
                'Persona Balance',
                'Energy Flow'
            )
        )
        
        # Add consciousness field surface
        x = y = z = np.linspace(-2, 2, 50)
        X, Y, Z = np.meshgrid(x, y, z)
        values = self._calculate_field_values(X, Y, Z)
        
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=values,
                colorscale=self.colors['consciousness']['field'],
                showscale=True,
                name='Consciousness Field'
            ),
            row=1, col=1
        )
        
        # Add sacred geometry patterns
        self._add_sacred_geometry(fig, row=1, col=2)
        
        # Add persona balance
        self._add_persona_balance(fig, row=2, col=1)
        
        # Add energy flow
        self._add_energy_flow(fig, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title='Hermes Consciousness Visualization',
            showlegend=True,
            height=1000,
            width=1200
        )
        
        return fig
        
    def _calculate_field_values(self, X: np.ndarray, Y: np.ndarray, 
                              Z: np.ndarray) -> np.ndarray:
        """Calculate consciousness field values."""
        values = np.zeros_like(X)
        
        # Get persona states
        metrics = self.personas.get_balance_metrics()
        apre_energy = metrics['apre_energy']
        magi_energy = metrics['magi_energy']
        
        # Calculate field values
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    # Distance from origin
                    r = np.sqrt(X[i,j,k]**2 + Y[i,j,k]**2 + Z[i,j,k]**2)
                    
                    # Base field
                    base = np.exp(-r/2)
                    
                    # Add persona influences
                    apre = apre_energy * np.cos(r * settings.sacred_ratios['phi'])
                    magi = magi_energy * np.sin(r * settings.sacred_ratios['phi'])
                    
                    # Combine
                    values[i,j,k] = base * (apre + magi)
                    
        # Add toroidal modulation
        toroidal = self.quantum.calculate_toroidal_field()
        values = values * np.abs(toroidal) * np.cos(np.angle(toroidal))
        
        return values
        
    def _add_sacred_geometry(self, fig: go.Figure, row: int, col: int):
        """Add sacred geometry visualization."""
        # Get Merkaba vertices
        merkaba = self.merkaba.get_pattern()
        
        # Add Merkaba
        for face in merkaba['upward'] + merkaba['downward']:
            points = face['points']
            x, y, z = points.T
            
            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='white', width=2),
                    name='Merkaba'
                ),
                row=row, col=col
            )
            
        # Add Sri Yantra
        sri = self.sri_yantra.get_consciousness_field()
        x = y = np.linspace(-1, 1, sri.shape[0])
        X, Y = np.meshgrid(x, y)
        
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=sri,
                colorscale=self.colors['consciousness']['sacred'],
                showscale=False,
                opacity=0.7,
                name='Sri Yantra'
            ),
            row=row, col=col
        )
        
    def _add_persona_balance(self, fig: go.Figure, row: int, col: int):
        """Add persona balance visualization."""
        # Get persona attributes
        attrs = self.personas.get_attribute_strengths()
        
        # Create radar chart for each persona
        for persona, attributes in attrs.items():
            theta = list(attributes.keys())
            r = list(attributes.values())
            
            fig.add_trace(
                go.Scatterpolar(
                    r=r + [r[0]],
                    theta=theta + [theta[0]],
                    fill='toself',
                    name=persona,
                    line_color=self.colors[persona.lower()]['primary']
                ),
                row=row, col=col
            )
            
    def _add_energy_flow(self, fig: go.Figure, row: int, col: int):
        """Add energy flow visualization."""
        # Get energy metrics
        metrics = self.personas.get_balance_metrics()
        
        # Create time series
        t = np.linspace(0, 2*np.pi, 100)
        
        # Apré energy flow
        apre_energy = metrics['apre_energy'] * np.cos(t * settings.sacred_ratios['phi'])
        fig.add_trace(
            go.Scatter(
                x=t, y=apre_energy,
                mode='lines',
                name='Apré Energy',
                line_color=self.colors['apre']['energy']
            ),
            row=row, col=col
        )
        
        # Magí energy flow
        magi_energy = metrics['magi_energy'] * np.sin(t * settings.sacred_ratios['phi'])
        fig.add_trace(
            go.Scatter(
                x=t, y=magi_energy,
                mode='lines',
                name='Magí Energy',
                line_color=self.colors['magi']['energy']
            ),
            row=row, col=col
        )
        
    def create_animation(self, duration: float = 10.0, fps: int = 30) -> go.Figure:
        """
        Create animated visualization of consciousness evolution.
        
        Args:
            duration: Animation duration in seconds
            fps: Frames per second
        """
        # Create frames
        frames = []
        n_frames = int(duration * fps)
        
        for i in range(n_frames):
            t = i / fps
            
            # Update personas
            self.personas.evolve_personas({'time': t})
            
            # Create frame
            frame = self.visualize_consciousness_field()
            frames.append(frame)
            
        # Create animation
        fig = frames[0]
        fig.frames = frames[1:]
        
        # Add slider
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 1000/fps, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate'
                    }]
                }]
            }],
            sliders=[{
                'currentvalue': {'prefix': 'Time: '},
                'steps': [
                    {
                        'args': [[f.name], {
                            'mode': 'immediate',
                            'frame': {'duration': 0, 'redraw': True},
                            'transition': {'duration': 0}
                        }],
                        'label': f'{i/fps:.1f}s',
                        'method': 'animate'
                    }
                    for i, f in enumerate(fig.frames)
                ]
            }]
        )
        
        return fig

# Global visualizer instance
consciousness_visualizer = ConsciousnessVisualizer()
