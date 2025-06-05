"""
BOLD5000 Visualization Utilities
This module provides tools to visualize and understand the BOLD5000 dataset,
helping us verify our data pipeline and gain insights into brain responses.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from PIL import Image
import warnings

# Set style for beautiful, publication-ready plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BOLD5000Visualizer:
    """
    Comprehensive visualization tools for BOLD5000 dataset exploration.
    
    Think of this as our 'microscope' for examining the data - it helps us
    see patterns and relationships that would be invisible in raw numbers.
    """
    
    def __init__(self, dataset, save_dir: str = 'visualizations'):
        """
        Initialize the visualizer with a dataset instance.
        
        Args:
            dataset: BOLD5000Dataset instance
            save_dir: Directory to save visualization outputs
        """
        self.dataset = dataset
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Cache ROI statistics for consistent scaling
        self.roi_stats = dataset.get_roi_statistics()
        
    def visualize_sample(self, idx: int, save_name: Optional[str] = None):
        """
        Create a comprehensive visualization of a single sample.
        
        This shows:
        - The image itself
        - RDM patterns for different ROIs
        - How this image relates to others in the session
        """
        # Get the sample with metadata
        self.dataset.return_metadata = True
        sample = self.dataset[idx]
        self.dataset.return_metadata = False
        
        # Create a figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot the image
        ax_img = fig.add_subplot(gs[0:2, 0:2])
        self._plot_image(sample['image'], ax_img)
        ax_img.set_title(f"Image: {Path(sample['metadata']['image_path']).stem}\n"
                         f"Subject: {sample['metadata']['subject']}, "
                         f"Session: {sample['metadata']['session']}, "
                         f"Trial: {sample['metadata']['trial_idx']}", fontsize=12)
        
        # Plot RDM patterns for key ROIs
        roi_positions = [
            ('LHEarlyVis', gs[0, 2], 'Early Visual (LH)'),
            ('RHEarlyVis', gs[0, 3], 'Early Visual (RH)'),
            ('LHLOC', gs[1, 2], 'LOC (LH)'),
            ('RHLOC', gs[1, 3], 'LOC (RH)'),
            ('LHPPA', gs[2, 0], 'PPA (LH)'),
            ('RHPPA', gs[2, 1], 'PPA (RH)'),
            ('LHOPA', gs[2, 2], 'OPA (LH)'),
            ('RHOPA', gs[2, 3], 'OPA (RH)')
        ]
        
        for roi, position, title in roi_positions:
            if roi in sample['rdms']:
                ax = fig.add_subplot(position)
                self._plot_rdm_row(sample['rdms'][roi], ax, title)
        
        plt.suptitle('BOLD5000 Sample Visualization', fontsize=16, y=0.98)
        
        if save_name:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def _plot_image(self, image_tensor: torch.Tensor, ax):
        """Plot an image tensor, handling normalization."""
        # Denormalize the image
        img = image_tensor.clone()
        
        # ImageNet normalization parameters
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Convert to numpy and transpose for matplotlib
        img_np = img.permute(1, 2, 0).numpy()
        
        ax.imshow(img_np)
        ax.axis('off')
    
    def _plot_rdm_row(self, rdm_row: torch.Tensor, ax, title: str):
        """
        Plot a single RDM row as a heatmap.
        
        This visualization helps us understand how similar/different this image
        is to all other images in the session according to this brain region.
        """
        rdm_np = rdm_row.numpy()
        
        # Create a heatmap
        im = ax.imshow(rdm_np.reshape(-1, 1).T, aspect='auto', cmap='viridis')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Other images in session')
        ax.set_yticks([])
        
        # Add a subtle indicator for the current image's position
        trial_idx = np.argmin(rdm_np)  # The image itself should have minimum distance
        ax.axvline(x=trial_idx, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def visualize_roi_relationships(self, sample_size: int = 100, save_name: Optional[str] = None):
        """
        Visualize how different ROIs relate to each other in their response patterns.
        
        This helps us understand which brain regions process information similarly,
        which will inform our model architecture decisions.
        """
        print(f"Analyzing ROI relationships using {sample_size} samples...")
        
        # Collect RDM samples from multiple images
        roi_responses = {roi: [] for roi in self.dataset.rois}
        
        # Sample random indices
        indices = np.random.choice(len(self.dataset), sample_size, replace=False)
        
        for idx in indices:
            sample = self.dataset[idx]
            for roi, rdm_row in sample['rdms'].items():
                # Take a subset of dissimilarities to make computation tractable
                roi_responses[roi].append(rdm_row[:50].numpy())  # First 50 dissimilarities
        
        # Compute correlations between ROI response patterns
        roi_names = list(roi_responses.keys())
        n_rois = len(roi_names)
        correlation_matrix = np.zeros((n_rois, n_rois))
        
        for i, roi1 in enumerate(roi_names):
            for j, roi2 in enumerate(roi_names):
                # Flatten and concatenate all responses
                responses1 = np.concatenate(roi_responses[roi1])
                responses2 = np.concatenate(roi_responses[roi2])
                
                # Compute correlation
                correlation_matrix[i, j] = np.corrcoef(responses1, responses2)[0, 1]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Correlation matrix
        sns.heatmap(correlation_matrix, 
                    xticklabels=roi_names,
                    yticklabels=roi_names,
                    annot=True,
                    fmt='.2f',
                    cmap='coolwarm',
                    center=0,
                    ax=ax1,
                    square=True)
        ax1.set_title('ROI Response Pattern Correlations', fontsize=14)
        
        # Hierarchical clustering visualization
        from scipy.cluster.hierarchy import dendrogram, linkage
        
        # Compute linkage
        linkage_matrix = linkage(1 - correlation_matrix, method='ward')
        
        # Create dendrogram
        dendro = dendrogram(linkage_matrix, labels=roi_names, ax=ax2)
        ax2.set_title('Hierarchical Clustering of ROIs', fontsize=14)
        ax2.set_xlabel('ROI')
        ax2.set_ylabel('Distance')
        
        plt.suptitle('ROI Relationship Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved ROI relationship analysis to {save_path}")
        
        plt.show()
        
        # Return the correlation matrix for further analysis
        return correlation_matrix, roi_names
    
    def visualize_image_diversity(self, n_images: int = 16, save_name: Optional[str] = None):
        """
        Show a diverse sample of images from the dataset.
        
        This helps us understand the visual diversity of stimuli used in BOLD5000,
        which spans natural scenes, objects, and various categories.
        """
        # Sample diverse indices
        indices = np.linspace(0, len(self.dataset)-1, n_images, dtype=int)
        
        # Create grid
        n_cols = 4
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            if i < len(axes):
                sample = self.dataset[idx]
                self._plot_image(sample['image'], axes[i])
                
                # Add image info
                if self.dataset.return_metadata:
                    name = Path(sample['metadata']['image_path']).stem
                    axes[i].set_title(f"{name[:20]}...", fontsize=8)
        
        # Hide empty subplots
        for i in range(len(indices), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Sample Images from BOLD5000 Dataset', fontsize=16)
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved image diversity visualization to {save_path}")
        
        plt.show()
    
    def plot_rdm_statistics(self, save_name: Optional[str] = None):
        """
        Visualize statistics about the RDMs across different ROIs.
        
        This gives us insights into which brain regions show more varied responses
        and helps us set appropriate weights for our multi-ROI loss function.
        """
        # Create subplots for different statistics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Prepare data
        rois = list(self.roi_stats.keys())
        means = [self.roi_stats[roi]['mean'] for roi in rois]
        stds = [self.roi_stats[roi]['std'] for roi in rois]
        mins = [self.roi_stats[roi]['min'] for roi in rois]
        maxs = [self.roi_stats[roi]['max'] for roi in rois]
        
        # Plot 1: Mean dissimilarity by ROI
        ax = axes[0]
        bars = ax.bar(range(len(rois)), means, color=sns.color_palette("husl", len(rois)))
        ax.set_xticks(range(len(rois)))
        ax.set_xticklabels(rois, rotation=45, ha='right')
        ax.set_ylabel('Mean Dissimilarity')
        ax.set_title('Average Dissimilarity by ROI')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Standard deviation by ROI
        ax = axes[1]
        bars = ax.bar(range(len(rois)), stds, color=sns.color_palette("husl", len(rois)))
        ax.set_xticks(range(len(rois)))
        ax.set_xticklabels(rois, rotation=45, ha='right')
        ax.set_ylabel('Standard Deviation')
        ax.set_title('Response Variability by ROI')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Range (max - min) by ROI
        ax = axes[2]
        ranges = [maxs[i] - mins[i] for i in range(len(rois))]
        bars = ax.bar(range(len(rois)), ranges, color=sns.color_palette("husl", len(rois)))
        ax.set_xticks(range(len(rois)))
        ax.set_xticklabels(rois, rotation=45, ha='right')
        ax.set_ylabel('Range (max - min)')
        ax.set_title('Dynamic Range by ROI')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Coefficient of variation (CV = std/mean)
        ax = axes[3]
        cvs = [stds[i]/means[i] if means[i] > 0 else 0 for i in range(len(rois))]
        bars = ax.bar(range(len(rois)), cvs, color=sns.color_palette("husl", len(rois)))
        ax.set_xticks(range(len(rois)))
        ax.set_xticklabels(rois, rotation=45, ha='right')
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('Relative Variability by ROI')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('RDM Statistics Across ROIs', fontsize=16)
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved RDM statistics to {save_path}")
        
        plt.show()


# Test the visualizer
if __name__ == "__main__":
    from src.data.bold5000_dataset import BOLD5000Dataset
    
    # Create a dataset for visualization
    print("Creating dataset for visualization...")
    dataset = BOLD5000Dataset(
        data_root='data/BOLD5000',
        subjects=['CSI1'],
        sessions=[1],
        return_metadata=True,
        verbose=False
    )
    
    # Create visualizer
    visualizer = BOLD5000Visualizer(dataset)
    
    # Run various visualizations
    print("\nGenerating visualizations...")
    
    # 1. Visualize a single sample
    print("1. Visualizing sample #42...")
    visualizer.visualize_sample(42, save_name='sample_42_visualization')
    
    # 2. Visualize ROI relationships
    print("\n2. Analyzing ROI relationships...")
    correlation_matrix, roi_names = visualizer.visualize_roi_relationships(
        sample_size=50, 
        save_name='roi_relationships'
    )
    
    # 3. Show image diversity
    print("\n3. Showing image diversity...")
    visualizer.visualize_image_diversity(n_images=16, save_name='image_diversity')
    
    # 4. Plot RDM statistics
    print("\n4. Plotting RDM statistics...")
    visualizer.plot_rdm_statistics(save_name='rdm_statistics')
    
    print("\nAll visualizations complete! Check the 'visualizations' directory.")