"""
BOLD5000 PyTorch Dataset
This module creates a PyTorch Dataset that understands the complex relationships
between images and brain responses. It's designed to support our goal of training
a neural network that mimics human visual processing.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Optional, Tuple, Union
import warnings
from collections import defaultdict

from .bold5000_metadata_enhanced import BOLD5000MetadataEnhanced


class BOLD5000Dataset(Dataset):
    """
    PyTorch Dataset for BOLD5000 that provides flexible access to images and brain responses.
    
    This dataset can operate in several modes:
    1. Image-only mode: Just load images (useful for pre-training)
    2. Single-subject mode: Load images with one subject's brain responses
    3. Multi-subject mode: Load images with multiple subjects' responses
    4. Session-aware mode: Respect the experimental session structure
    
    The flexibility allows us to start simple and gradually increase complexity.
    """
    
    def __init__(
        self,
        data_root: str,
        subjects: Union[str, List[str]] = 'all',
        rois: Optional[List[str]] = None,
        sessions: Optional[List[int]] = None,
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        return_metadata: bool = False,
        aggregation: str = 'mean',
        verbose: bool = True
    ):
        """
        Initialize the BOLD5000 dataset.
        
        Args:
            data_root: Path to BOLD5000 root directory
            subjects: Which subjects to include ('all', single subject, or list)
            rois: Which ROIs to load (None = all available)
            sessions: Which sessions to include (None = all)
            image_size: Size to resize images to
            transform: Optional torchvision transforms
            return_metadata: Whether to return detailed metadata with each sample
            aggregation: How to aggregate multiple viewings ('mean', 'first', 'all')
            verbose: Whether to print loading information
        """
        self.data_root = Path(data_root)
        self.verbose = verbose
        self.return_metadata = return_metadata
        self.aggregation = aggregation
        self.image_size = image_size
        
        # Load metadata using our enhanced parser
        if self.verbose:
            print("Initializing BOLD5000 Dataset...")
        self.metadata = BOLD5000MetadataEnhanced(data_root, verbose=False)
        
        # Process subject selection
        self.subjects = self._process_subject_selection(subjects)
        
        # Process ROI selection
        self.rois = rois if rois is not None else self.metadata.rois
        
        # Process session selection
        self.sessions = self._process_session_selection(sessions)
        
        # Build the dataset index
        self._build_dataset_index()
        
        # Setup image transforms
        self.transform = transform if transform is not None else self._get_default_transform()
        
        if self.verbose:
            print(f"\nDataset initialized:")
            print(f"  Total samples: {len(self)}")
            print(f"  Subjects: {', '.join(self.subjects)}")
            print(f"  ROIs: {len(self.rois)}")
            print(f"  Sessions: {len(self.sessions)} per subject")
    
    def _process_subject_selection(self, subjects: Union[str, List[str]]) -> List[str]:
        """Process subject selection input"""
        if subjects == 'all':
            return self.metadata.subjects
        elif isinstance(subjects, str):
            return [subjects]
        else:
            return subjects
    
    def _process_session_selection(self, sessions: Optional[List[int]]) -> Dict[str, List[int]]:
        """Process session selection for each subject"""
        session_dict = {}
        
        for subject in self.subjects:
            # Get available sessions for this subject
            available = len(self.metadata.session_boundaries.get(subject, []))
            
            if sessions is None:
                # Use all available sessions
                session_dict[subject] = list(range(1, available + 1))
            else:
                # Use only requested sessions that exist
                session_dict[subject] = [s for s in sessions if 1 <= s <= available]
                
        return session_dict
    
    def _build_dataset_index(self):
        """
        Build an index of all samples in the dataset.
        This is where we decide what constitutes a "sample" for training.
        """
        self.samples = []
        self.image_to_samples = defaultdict(list)  # Track which samples use which image
        
        if self.verbose:
            print("\nBuilding dataset index...")
        
        # Iterate through all subjects and sessions
        for subject in self.subjects:
            for session in self.sessions.get(subject, []):
                # Get session data
                session_data = self.metadata.get_session_data(subject, session)
                
                # Check if we have the required RDMs
                available_rois = set(session_data['available_rdms'].keys())
                required_rois = set(self.rois)
                
                if not required_rois.issubset(available_rois):
                    missing = required_rois - available_rois
                    if self.verbose:
                        print(f"  Skipping {subject} session {session}: missing ROIs {missing}")
                    continue
                
                # Add each trial as a potential sample
                for trial_info in session_data['images']:
                    sample = {
                        'subject': subject,
                        'session': session,
                        'trial_idx': trial_info['trial_index'],
                        'image_path': trial_info['image_path'],
                        'presentation_name': trial_info['presentation_name'],
                        'rdm_paths': {roi: session_data['available_rdms'][roi] 
                                     for roi in self.rois}
                    }
                    
                    self.samples.append(sample)
                    self.image_to_samples[str(trial_info['image_path'])].append(len(self.samples) - 1)
        
        if self.verbose:
            print(f"  Built index with {len(self.samples)} samples")
            print(f"  Unique images: {len(self.image_to_samples)}")
    
    def _get_default_transform(self):
        """Get default image transformation pipeline"""
        # This transformation pipeline is carefully designed for brain-alignment
        # We want to preserve as much information as possible while standardizing size
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            # Normalize using ImageNet statistics since BOLD5000 includes ImageNet images
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        """Return the total number of samples"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Returns a dictionary containing:
        - 'image': The preprocessed image tensor
        - 'rdms': Dictionary of RDM rows for this image in each ROI
        - 'metadata': (optional) Additional information about the sample
        """
        sample_info = self.samples[idx]
        
        # Load and preprocess the image
        image = Image.open(sample_info['image_path']).convert('RGB')
        image_tensor = self.transform(image)
        
        # Load RDM data for this trial
        rdm_data = {}
        for roi, rdm_path in sample_info['rdm_paths'].items():
            # Load the full RDM for this session
            full_rdm = np.load(rdm_path)
            
            # Extract the row corresponding to this trial
            # RDMs are symmetric, so we can use either row or column
            trial_idx = sample_info['trial_idx']
            rdm_row = full_rdm[trial_idx, :]
            
            # Convert to tensor
            rdm_data[roi] = torch.from_numpy(rdm_row).float()
        
        # Prepare the return dictionary
        sample = {
            'image': image_tensor,
            'rdms': rdm_data
        }
        
        # Add metadata if requested
        if self.return_metadata:
            sample['metadata'] = {
                'subject': sample_info['subject'],
                'session': sample_info['session'],
                'trial_idx': sample_info['trial_idx'],
                'image_path': str(sample_info['image_path']),
                'presentation_name': sample_info['presentation_name']
            }
        
        return sample
    
    def get_image_aggregated_responses(self, image_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
        """
        Get aggregated brain responses for a specific image across all its presentations.
        
        This is useful for getting a stable estimate of brain responses by averaging
        across multiple viewings of the same image.
        """
        image_path = str(image_path)
        sample_indices = self.image_to_samples.get(image_path, [])
        
        if not sample_indices:
            raise ValueError(f"Image {image_path} not found in dataset")
        
        # Collect all RDM rows for this image
        all_rdms = defaultdict(list)
        
        for idx in sample_indices:
            sample = self[idx]
            for roi, rdm_row in sample['rdms'].items():
                all_rdms[roi].append(rdm_row)
        
        # Aggregate according to strategy
        aggregated_rdms = {}
        for roi, rdm_list in all_rdms.items():
            rdm_stack = torch.stack(rdm_list)
            
            if self.aggregation == 'mean':
                aggregated_rdms[roi] = rdm_stack.mean(dim=0)
            elif self.aggregation == 'first':
                aggregated_rdms[roi] = rdm_stack[0]
            elif self.aggregation == 'all':
                aggregated_rdms[roi] = rdm_stack
            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        return aggregated_rdms
    
    def get_roi_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics about the RDMs for each ROI.
        This helps us understand the data distribution and set appropriate loss weights.
        """
        if self.verbose:
            print("\nComputing ROI statistics...")
        
        roi_stats = defaultdict(lambda: {'mean': 0, 'std': 0, 'min': float('inf'), 'max': float('-inf')})
        roi_values = defaultdict(list)
        
        # Sample a subset of the dataset for efficiency
        sample_size = min(1000, len(self))
        indices = np.random.choice(len(self), sample_size, replace=False)
        
        for idx in indices:
            sample = self[idx]
            for roi, rdm_row in sample['rdms'].items():
                # We typically look at the upper triangle of RDMs (excluding diagonal)
                # But since we have rows, we'll compute stats on the full row
                values = rdm_row.numpy()
                roi_values[roi].extend(values.tolist())
        
        # Compute statistics
        for roi, values in roi_values.items():
            values = np.array(values)
            roi_stats[roi] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        return dict(roi_stats)


# Utility function to create data loaders
def create_bold5000_loaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_subjects: List[str] = ['CSI1', 'CSI2', 'CSI3'],
    val_subjects: List[str] = ['CSI4'],
    train_sessions: Optional[List[int]] = None,
    val_sessions: Optional[List[int]] = None,
    **dataset_kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation data loaders with proper subject splits.
    
    The default split uses CSI1-3 for training and CSI4 for validation,
    which is reasonable since CSI4 saw fewer images.
    """
    # Create datasets
    train_dataset = BOLD5000Dataset(
        data_root=data_root,
        subjects=train_subjects,
        sessions=train_sessions,
        **dataset_kwargs
    )
    
    val_dataset = BOLD5000Dataset(
        data_root=data_root,
        subjects=val_subjects,
        sessions=val_sessions,
        **dataset_kwargs
    )
    
    # Create loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


# Test the dataset
if __name__ == "__main__":
    # Create a small test dataset
    dataset = BOLD5000Dataset(
        data_root='data/BOLD5000',
        subjects=['CSI1'],
        sessions=[1],
        return_metadata=True,
        verbose=True
    )
    
    # Test loading a sample
    print("\nTesting sample loading...")
    sample = dataset[0]
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  RDMs available: {list(sample['rdms'].keys())}")
    print(f"  RDM shape (per ROI): {sample['rdms']['LHEarlyVis'].shape}")
    
    # Test ROI statistics
    stats = dataset.get_roi_statistics()
    print("\nROI Statistics (sample):")
    for roi in ['LHEarlyVis', 'LHPPA', 'LHLOC']:
        if roi in stats:
            print(f"  {roi}: mean={stats[roi]['mean']:.3f}, std={stats[roi]['std']:.3f}")