"""
Field rendering and visualization tools for sacred geometry patterns.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import torch
from loguru import logger
from ..sacred.patterns import Merkaba
from ..quantum.error_correction import QuantumErrorCorrector

class FieldVisualizer:
    """
    Visualization system for electromagnetic and consciousness fields.
    Optimized for M1 GPU acceleration.
    """
    
    def __init__(self, use_gpu: bool = True):
        """Initialize visualizer with GPU support if available."""
        self.device = torch.device("mps" if use_gpu and torch.backends.mps.is_available() else "cpu")
        self.cache = {}
        logger.info(f"FieldVisualizer initialized on device: {self.device}")
        
    def _prepare_field_data(self, field: np.ndarray) -> torch.Tensor:
        """Convert numpy array to GPU tensor if available."""
        return torch.from_numpy(field).to(self.device)
        
    def plot_merkaba_fields(self, merkaba: Merkaba,
                           resolution: int = 32,
                           opacity: float = 0.1) -> go.Figure:
        """
        Create 3D visualization of Merkaba fields.
        
        Args:
            merkaba: Merkaba instance to visualize
            resolution: Grid resolution (default: 32 for M1 optimization)
            opacity: Opacity of volume rendering
            
        Returns:
            Plotly figure object
        """
        # Convert fields to tensors
        mag_field = self._prepare_field_data(merkaba.magnetic_field)
        elec_field = self._prepare_field_data(merkaba.electric_field)
        
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=('Magnetic Field', 'Electric Field')
        )
        
        # Grid coordinates
        coords = np.linspace(-1, 1, resolution)
        X, Y, Z = np.meshgrid(coords, coords, coords)
        
        # Add magnetic field visualization
        fig.add_trace(
            go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=mag_field.cpu().numpy().flatten(),
                isomin=0.1,
                isomax=1.0,
                opacity=opacity,
                surface_count=20,
                colorscale='Viridis',
                colorbar=dict(title='Magnetic Field Strength')
            ),
            row=1, col=1
        )
        
        # Add electric field visualization
        fig.add_trace(
            go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=elec_field.cpu().numpy().flatten(),
                isomin=0.1,
                isomax=1.0,
                opacity=opacity,
                surface_count=20,
                colorscale='Plasma',
                colorbar=dict(title='Electric Field Strength')
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Merkaba Consciousness Fields',
            scene=dict(
                xaxis_title='X (Consciousness)',
                yaxis_title='Y (Harmonic)',
                zaxis_title='Z (Temporal)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            scene2=dict(
                xaxis_title='X (Consciousness)',
                yaxis_title='Y (Harmonic)',
                zaxis_title='Z (Temporal)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            )
        )
        
        return fig
        
    def plot_consciousness_evolution(self, merkaba: Merkaba,
                                  time_steps: int = 100) -> go.Figure:
        """
        Visualize consciousness evolution over time.
        
        Args:
            merkaba: Merkaba instance
            time_steps: Number of evolution steps
            
        Returns:
            Plotly figure with consciousness metrics
        """
        times = np.linspace(0, 10, time_steps)
        consciousness_levels = []
        field_strengths = []
        
        # Calculate evolution
        for t in times:
            merkaba.rotate(t)
            consciousness_levels.append(merkaba.get_consciousness_level(t))
            field_strengths.append(
                np.mean(np.abs(merkaba.magnetic_field)) +
                np.mean(np.abs(merkaba.electric_field))
            )
            
        # Create figure
        fig = go.Figure()
        
        # Add consciousness level trace
        fig.add_trace(go.Scatter(
            x=times,
            y=consciousness_levels,
            name='Consciousness Level',
            line=dict(color='purple', width=2)
        ))
        
        # Add field strength trace
        fig.add_trace(go.Scatter(
            x=times,
            y=field_strengths,
            name='Field Strength',
            line=dict(color='blue', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title='Consciousness Evolution',
            xaxis_title='Time',
            yaxis_title='Magnitude',
            template='plotly_dark'
        )
        
        return fig
        
    def plot_error_correction(self, corrector: QuantumErrorCorrector,
                            state_vector: np.ndarray) -> go.Figure:
        """
        Visualize quantum error correction process.
        
        Args:
            corrector: Quantum error corrector instance
            state_vector: Input quantum state
            
        Returns:
            Plotly figure showing error correction
        """
        # Get error correction steps
        syndrome = corrector.measure_syndrome(state_vector)
        corrected_state = corrector.correct_errors(state_vector, syndrome)
        
        # Create figure
        fig = go.Figure()
        
        # Plot original state
        fig.add_trace(go.Bar(
            x=['|0⟩', '|1⟩'],
            y=np.abs(state_vector)**2,
            name='Original State',
            marker_color='blue'
        ))
        
        # Plot corrected state
        fig.add_trace(go.Bar(
            x=['|0⟩', '|1⟩'],
            y=np.abs(corrected_state)**2,
            name='Corrected State',
            marker_color='green'
        ))
        
        # Update layout
        fig.update_layout(
            title='Quantum Error Correction',
            xaxis_title='Basis States',
            yaxis_title='Probability',
            barmode='group',
            template='plotly_dark'
        )
        
        return fig
        
    def save_animation(self, merkaba: Merkaba,
                      filename: str,
                      duration: float = 10.0,
                      fps: int = 30) -> None:
        """
        Create and save an animation of rotating Merkaba fields.
        
        Args:
            merkaba: Merkaba instance
            filename: Output filename
            duration: Animation duration in seconds
            fps: Frames per second
        """
        import plotly.io as pio
        
        frames = []
        n_frames = int(duration * fps)
        
        for i in range(n_frames):
            t = i / fps
            merkaba.rotate(t)
            fig = self.plot_merkaba_fields(merkaba)
            frames.append(fig.frames[0])
            
        # Create animation
        fig = go.Figure(
            data=frames[0].data,
            layout=frames[0].layout,
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 1000/fps, 'redraw': True},
                                      'fromcurrent': True}],
                        'label': '▶️',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0, 'redraw': True},
                                        'mode': 'immediate',
                                        'transition': {'duration': 0}}],
                        'label': '⏸️',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }]
        )
        
        # Save animation
        pio.write_html(fig, filename)
