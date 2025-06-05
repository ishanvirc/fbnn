"""
BOLD5000 Metadata Parser
This module handles the complex relationships between images, subjects, sessions,
and brain responses in the BOLD5000 dataset. Think of it as the "librarian" that
knows where everything is and how it all connects.
"""

import numpy as np
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
import re
from collections import defaultdict
import json


class BOLD5000Metadata:
    """
    Central metadata manager for BOLD5000 dataset.
    
    This class understands:
    - Which images were shown to which subjects
    - The presentation order within sessions
    - Mapping between image files and brain responses
    - ROI specifications and naming conventions
    """
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.subjects = ['CSI1', 'CSI2', 'CSI3', 'CSI4']
        self.rois = self._get_roi_names()
        self.sessions_per_subject = 15
        
        # These will be populated by parsing
        self.image_paths = {}  # image_id -> full path
        self.subject_image_lists = {}  # subject -> list of image names in order
        self.image_to_trials = {}  # (subject, image) -> list of (session, trial_idx)
        self.trial_to_image = {}  # (subject, session, trial_idx) -> image_id
        
        # Parse all metadata
        self._parse_metadata()
        
    def _get_roi_names(self) -> List[str]:
        """Get the standard ROI names used in the dataset"""
        # Based on what we found in the explorer
        return [
            'LHEarlyVis', 'RHEarlyVis',
            'LHLOC', 'RHLOC',
            'LHOPA', 'RHOPA', 
            'LHPPA', 'RHPPA',
            'LHRSC', 'RHRSC'  # Note: The data shows 'RHRRSC' but that's likely a typo
        ]
    
    def _parse_metadata(self):
        """Parse all metadata files to build our mappings"""
        print("Parsing BOLD5000 metadata...")
        
        # First, catalog all images
        self._catalog_images()
        
        # Then parse subject-specific presentation orders
        for subject in self.subjects:
            self._parse_subject_metadata(subject)
            
        # Create reverse mappings for efficient lookup
        self._create_reverse_mappings()
        
        print(f"Metadata parsing complete!")
        print(f"  Total unique images: {len(self.image_paths)}")
        print(f"  Subjects with data: {len(self.subject_image_lists)}")
        
    def _catalog_images(self):
        """Build a catalog of all available images"""
        # Main stimulus directories to check
        stim_dirs = [
            self.data_root / 'data' / 'images' / 'BOLD5000_Stimuli' / 'Scene_Stimuli' / 'Presented_Stimuli',
            self.data_root / 'data' / 'images' / 'BOLD5000_Stimuli' / 'Scene_Stimuli' / 'Original_Images'
        ]
        
        image_count = 0
        for stim_dir in stim_dirs:
            if stim_dir.exists():
                # Find all images recursively
                for img_path in stim_dir.rglob('*.jpg'):
                    # Use the filename as the image ID
                    img_id = img_path.stem  # filename without extension
                    self.image_paths[img_id] = img_path
                    image_count += 1
                    
        print(f"  Cataloged {image_count} images")
        
    def _parse_subject_metadata(self, subject: str):
        """Parse metadata for a specific subject"""
        # Look for the subject's image list
        imgnames_path = (self.data_root / 'data' / 'images' / 'BOLD5000_Stimuli' / 
                        'Stimuli_Presentation_Lists' / subject / f'{subject}_imgnames.txt')
        
        if not imgnames_path.exists():
            print(f"  Warning: No image list found for {subject}")
            return
            
        # Read the image presentation order
        with open(imgnames_path, 'r') as f:
            lines = f.readlines()
            
        # Parse the presentation order
        # The file typically contains image names, possibly with session/run info
        image_list = []
        session_trial_mapping = defaultdict(list)
        
        for idx, line in enumerate(lines):
            line = line.strip()
            if line and not line.startswith('#'):  # Skip comments
                # Extract image name (might need adjustment based on actual format)
                # Sometimes includes path info we need to strip
                if '/' in line:
                    img_name = Path(line).stem
                else:
                    img_name = line.replace('.jpg', '').replace('.JPEG', '')
                    
                image_list.append(img_name)
                
                # Calculate which session and trial this is
                # Assuming ~370 trials per session based on RDM sizes
                session_num = (idx // 370) + 1
                trial_in_session = idx % 370
                
                # Store the mapping
                key = (subject, img_name)
                if key not in self.image_to_trials:
                    self.image_to_trials[key] = []
                self.image_to_trials[key].append((session_num, trial_in_session))
                
                # Store reverse mapping
                self.trial_to_image[(subject, session_num, trial_in_session)] = img_name
                
        self.subject_image_lists[subject] = image_list
        print(f"  Parsed {len(image_list)} presentations for {subject}")
        
    def _create_reverse_mappings(self):
        """Create additional mappings for efficient data access"""
        # Create a mapping of which subjects saw which images
        self.image_to_subjects = defaultdict(list)
        for subject, img_list in self.subject_image_lists.items():
            for img in set(img_list):  # unique images only
                self.image_to_subjects[img].append(subject)
                
    def get_rdm_path(self, subject: str, session: int, roi: str, 
                     run: Optional[int] = None) -> Path:
        """
        Get the path to an RDM file.
        
        Args:
            subject: Subject ID (e.g., 'CSI1')
            session: Session number (1-15)
            roi: ROI name (e.g., 'LHEarlyVis')
            run: Optional run number (1-10). If None, returns full session RDM
            
        Returns:
            Path to the RDM .npy file
        """
        base_path = self.data_root / 'analysis' / 'fMRI_RDMs' / subject / f'session_{session}'
        
        if run is None:
            # Full session RDM
            rdm_path = (base_path / 'ROI_specific' / roi / 'full_session' / 
                       f'{subject}_sess{session}_{roi}_fmri_rdm.npy')
        else:
            # Run-specific RDM
            rdm_path = (base_path / 'ROI_specific' / roi / f'run_{run:02d}' /
                       f'{subject}_sess{session}_{roi}_run{run:02d}_fmri_rdm.npy')
            
        return rdm_path
    
    def get_image_info(self, image_id: str) -> Dict:
        """Get comprehensive information about an image"""
        info = {
            'image_id': image_id,
            'path': self.image_paths.get(image_id),
            'subjects_viewed': self.image_to_subjects.get(image_id, []),
            'presentations': {}
        }
        
        # For each subject that saw this image, list when
        for subject in info['subjects_viewed']:
            key = (subject, image_id)
            if key in self.image_to_trials:
                info['presentations'][subject] = self.image_to_trials[key]
                
        return info
    
    def get_session_info(self, subject: str, session: int) -> Dict:
        """Get information about a specific session"""
        info = {
            'subject': subject,
            'session': session,
            'num_trials': 0,
            'images': [],
            'available_rdms': {}
        }
        
        # Find all images in this session
        session_images = []
        for trial_idx in range(400):  # Check up to 400 trials
            key = (subject, session, trial_idx)
            if key in self.trial_to_image:
                session_images.append(self.trial_to_image[key])
                
        info['num_trials'] = len(session_images)
        info['images'] = session_images
        
        # Check which RDMs exist for this session
        for roi in self.rois:
            rdm_path = self.get_rdm_path(subject, session, roi)
            if rdm_path.exists():
                info['available_rdms'][roi] = rdm_path
                
        return info
    
    def validate_data_integrity(self) -> Dict:
        """Run integrity checks on the dataset"""
        report = {
            'missing_images': [],
            'missing_rdms': [],
            'rdm_size_mismatches': []
        }
        
        print("\nValidating data integrity...")
        
        # Check if all referenced images exist
        for img_id in self.image_to_subjects.keys():
            if img_id not in self.image_paths:
                report['missing_images'].append(img_id)
                
        # Check RDM availability and sizes
        for subject in self.subjects:
            for session in range(1, 16):
                session_info = self.get_session_info(subject, session)
                expected_size = session_info['num_trials']
                
                for roi in self.rois:
                    rdm_path = self.get_rdm_path(subject, session, roi)
                    if not rdm_path.exists():
                        report['missing_rdms'].append(f"{subject}_sess{session}_{roi}")
                    else:
                        # Check RDM size matches number of trials
                        rdm = np.load(rdm_path)
                        if rdm.shape[0] != expected_size:
                            report['rdm_size_mismatches'].append({
                                'file': rdm_path.name,
                                'expected': expected_size,
                                'actual': rdm.shape[0]
                            })
                            
        return report


# Test the metadata parser
if __name__ == "__main__":
    metadata = BOLD5000Metadata('data/BOLD5000')
    
    # Test getting info about a session
    session_info = metadata.get_session_info('CSI1', 1)
    print(f"\nSession 1 for CSI1:")
    print(f"  Number of trials: {session_info['num_trials']}")
    print(f"  Number of ROIs with RDMs: {len(session_info['available_rdms'])}")
    
    # Run validation
    validation_report = metadata.validate_data_integrity()
    if validation_report['missing_images']:
        print(f"\n⚠ Missing {len(validation_report['missing_images'])} images")
    if validation_report['missing_rdms']:
        print(f"⚠ Missing {len(validation_report['missing_rdms'])} RDMs")