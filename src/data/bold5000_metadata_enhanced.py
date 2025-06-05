"""
Enhanced BOLD5000 Metadata Parser
This version handles the real-world messiness of the dataset with more sophisticated
name matching and provides detailed debugging information.
"""

import numpy as np
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
import re
from collections import defaultdict
import json
from difflib import SequenceMatcher
import warnings


class BOLD5000MetadataEnhanced:
    """
    Enhanced metadata manager that handles naming inconsistencies gracefully.
    
    Think of this as a more forgiving librarian who can find books even when
    you don't remember the exact title.
    """
    
    def __init__(self, data_root: str, verbose: bool = True):
        self.data_root = Path(data_root)
        self.verbose = verbose
        self.subjects = ['CSI1', 'CSI2', 'CSI3', 'CSI4']
        self.rois = self._get_roi_names()
        
        # Image mappings with multiple strategies
        self.image_paths = {}  # normalized_name -> full path
        self.image_name_variants = defaultdict(set)  # track all variants of each image name
        self.presentation_to_image = {}  # (subject, presentation_name) -> actual image path
        self.missing_images = defaultdict(list)  # track what we couldn't find
        
        # Parse everything
        self._parse_metadata()
        
    def _get_roi_names(self) -> List[str]:
        """Get the ROI names, fixing the RHRRSC typo"""
        roi_names = [
            'LHEarlyVis', 'RHEarlyVis',
            'LHLOC', 'RHLOC',
            'LHOPA', 'RHOPA', 
            'LHPPA', 'RHPPA',
            'LHRSC', 'RHRSC'
        ]
        # Also check for the typo version
        self.roi_name_mapping = {'RHRRSC': 'RHRSC'}  # Fix typo if found
        return roi_names
    
    def _normalize_image_name(self, name: str) -> str:
        """
        Normalize an image name for matching.
        This handles various naming inconsistencies in BOLD5000.
        """
        # Remove path components
        name = Path(name).stem
        
        # Remove common prefixes/suffixes
        name = re.sub(r'^(rep_)?', '', name)  # Remove 'rep_' prefix
        name = re.sub(r'_\d+x\d+$', '', name)  # Remove resolution suffixes like '_375x500'
        
        # Standardize separators
        name = name.replace('-', '_')
        
        # Remove file extensions that might be embedded
        name = name.replace('.JPEG', '').replace('.jpg', '').replace('.png', '')
        
        return name.lower()
    
    def _catalog_images_smart(self):
        """
        Catalog images with multiple naming strategies.
        This is like creating a card catalog with cross-references.
        """
        if self.verbose:
            print("\nCataloging images with smart matching...")
            
        # All possible image locations
        image_dirs = [
            self.data_root / 'data' / 'images' / 'BOLD5000_Stimuli' / 'Scene_Stimuli' / 'Presented_Stimuli',
            self.data_root / 'data' / 'images' / 'BOLD5000_Stimuli' / 'Scene_Stimuli' / 'Original_Images',
            self.data_root / 'data' / 'images'  # Check root images dir too
        ]
        
        image_count = 0
        for img_dir in image_dirs:
            if img_dir.exists():
                for img_path in img_dir.rglob('*.jpg'):
                    image_count += 1
                    
                    # Store with multiple keys for flexible matching
                    full_name = img_path.stem
                    normalized_name = self._normalize_image_name(full_name)
                    
                    # Store the image with its normalized name
                    self.image_paths[normalized_name] = img_path
                    
                    # Keep track of all name variants we've seen
                    self.image_name_variants[normalized_name].add(full_name)
                    self.image_name_variants[normalized_name].add(img_path.name)
                    
                # Also check for JPEG files (different extension)
                for img_path in img_dir.rglob('*.JPEG'):
                    image_count += 1
                    full_name = img_path.stem
                    normalized_name = self._normalize_image_name(full_name)
                    self.image_paths[normalized_name] = img_path
                    self.image_name_variants[normalized_name].add(full_name)
                    
        if self.verbose:
            print(f"  Cataloged {image_count} total image files")
            print(f"  Normalized to {len(self.image_paths)} unique images")
            
    def _parse_subject_presentations(self, subject: str) -> List[str]:
        """
        Parse presentation order for a subject with better error handling.
        Returns the list of image names as they appear in the presentation file.
        """
        imgnames_path = (self.data_root / 'data' / 'images' / 'BOLD5000_Stimuli' / 
                        'Stimuli_Presentation_Lists' / subject / f'{subject}_imgnames.txt')
        
        if not imgnames_path.exists():
            warnings.warn(f"No image list found for {subject}")
            return []
            
        presentation_names = []
        with open(imgnames_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Keep the original presentation name for debugging
                    presentation_names.append(line)
                    
        return presentation_names
    
    def _match_presentation_to_image(self, presentation_name: str) -> Optional[Path]:
        """
        Try to match a presentation name to an actual image file.
        Uses multiple strategies from exact matching to fuzzy matching.
        """
        # Strategy 1: Direct normalized match
        normalized = self._normalize_image_name(presentation_name)
        if normalized in self.image_paths:
            return self.image_paths[normalized]
            
        # Strategy 2: Try without common category prefixes
        # BOLD5000 sometimes includes category names in presentations
        for prefix in ['COCO_', 'ImageNet_', 'Scene_', 'SUN_']:
            if presentation_name.startswith(prefix):
                shortened = presentation_name[len(prefix):]
                normalized_short = self._normalize_image_name(shortened)
                if normalized_short in self.image_paths:
                    return self.image_paths[normalized_short]
                    
        # Strategy 3: Fuzzy matching for close matches
        # This handles minor typos or formatting differences
        best_match = None
        best_score = 0.8  # Minimum similarity threshold
        
        for norm_name, path in self.image_paths.items():
            score = SequenceMatcher(None, normalized, norm_name).ratio()
            if score > best_score:
                best_score = score
                best_match = path
                
        return best_match
    
    def _parse_metadata(self):
        """Complete metadata parsing with improved matching"""
        # First catalog all images
        self._catalog_images_smart()
        
        # Parse each subject's data
        self.subject_presentations = {}
        self.session_boundaries = {}
        
        for subject in self.subjects:
            if self.verbose:
                print(f"\nParsing {subject} metadata...")
                
            presentation_names = self._parse_subject_presentations(subject)
            self.subject_presentations[subject] = presentation_names
            
            # Match presentations to actual images
            matched = 0
            unmatched = []
            
            for pres_name in presentation_names:
                image_path = self._match_presentation_to_image(pres_name)
                if image_path:
                    self.presentation_to_image[(subject, pres_name)] = image_path
                    matched += 1
                else:
                    unmatched.append(pres_name)
                    self.missing_images[subject].append(pres_name)
                    
            if self.verbose:
                print(f"  Presentations: {len(presentation_names)}")
                print(f"  Successfully matched: {matched} ({matched/len(presentation_names)*100:.1f}%)")
                if unmatched and self.verbose:
                    print(f"  Sample unmatched: {unmatched[:3]}...")
                    
            # Determine session boundaries
            # Each session is approximately 370 trials (from RDM analysis)
            self.session_boundaries[subject] = []
            trials_per_session = 370
            num_sessions = (len(presentation_names) + trials_per_session - 1) // trials_per_session
            
            for sess in range(num_sessions):
                start_idx = sess * trials_per_session
                end_idx = min((sess + 1) * trials_per_session, len(presentation_names))
                self.session_boundaries[subject].append((start_idx, end_idx))
                
    def get_session_data(self, subject: str, session: int) -> Dict:
        """
        Get all data for a specific session, with graceful handling of missing data.
        
        Returns:
            Dictionary with 'images', 'rdms', and 'missing' information
        """
        if subject not in self.subject_presentations:
            raise ValueError(f"Subject {subject} not found")
            
        if session < 1 or session > len(self.session_boundaries[subject]):
            raise ValueError(f"Session {session} out of range for {subject}")
            
        # Get the presentation indices for this session
        start_idx, end_idx = self.session_boundaries[subject][session - 1]
        session_presentations = self.subject_presentations[subject][start_idx:end_idx]
        
        # Collect available images
        available_images = []
        missing_images = []
        
        for pres_name in session_presentations:
            key = (subject, pres_name)
            if key in self.presentation_to_image:
                available_images.append({
                    'presentation_name': pres_name,
                    'image_path': self.presentation_to_image[key],
                    'trial_index': len(available_images)
                })
            else:
                missing_images.append(pres_name)
                
        # Check for RDMs
        available_rdms = {}
        for roi in self.rois:
            # Check both possible names (handling RHRRSC typo)
            roi_to_check = roi
            if roi == 'RHRSC':
                # Also check for the typo version
                alt_path = self.get_rdm_path(subject, session, 'RHRRSC')
                if alt_path.exists():
                    available_rdms[roi] = alt_path
                    continue
                    
            rdm_path = self.get_rdm_path(subject, session, roi)
            if rdm_path.exists():
                available_rdms[roi] = rdm_path
                
        return {
            'subject': subject,
            'session': session,
            'num_trials': len(session_presentations),
            'num_available_images': len(available_images),
            'images': available_images,
            'missing_images': missing_images,
            'available_rdms': available_rdms
        }
    
    def get_rdm_path(self, subject: str, session: int, roi: str, 
                     run: Optional[int] = None) -> Path:
        """Get RDM path, handling naming inconsistencies"""
        base_path = self.data_root / 'analysis' / 'fMRI_RDMs' / subject / f'session_{session}'
        
        if run is None:
            rdm_path = (base_path / 'ROI_specific' / roi / 'full_session' / 
                       f'{subject}_sess{session}_{roi}_fmri_rdm.npy')
        else:
            rdm_path = (base_path / 'ROI_specific' / roi / f'run_{run:02d}' /
                       f'{subject}_sess{session}_{roi}_run{run:02d}_fmri_rdm.npy')
            
        return rdm_path
    
    def print_summary_report(self):
        """Print a comprehensive summary of the dataset status"""
        print("\n" + "="*60)
        print("BOLD5000 Dataset Summary Report")
        print("="*60)
        
        total_presentations = sum(len(pres) for pres in self.subject_presentations.values())
        total_matched = len(self.presentation_to_image)
        
        print(f"\nOverall Statistics:")
        print(f"  Total presentations across all subjects: {total_presentations}")
        print(f"  Successfully matched to images: {total_matched} ({total_matched/total_presentations*100:.1f}%)")
        print(f"  Unique images in dataset: {len(self.image_paths)}")
        
        print(f"\nPer-Subject Breakdown:")
        for subject in self.subjects:
            pres_count = len(self.subject_presentations.get(subject, []))
            missing_count = len(self.missing_images.get(subject, []))
            matched_count = pres_count - missing_count
            
            print(f"\n  {subject}:")
            print(f"    Total presentations: {pres_count}")
            print(f"    Matched images: {matched_count} ({matched_count/pres_count*100:.1f}% success rate)")
            print(f"    Sessions: {len(self.session_boundaries.get(subject, []))}")
            
        # Check RDM coverage
        print(f"\nRDM Coverage Analysis:")
        rdm_coverage = defaultdict(int)
        total_possible_rdms = 0
        
        for subject in self.subjects:
            for session in range(1, 16):
                if session <= len(self.session_boundaries.get(subject, [])):
                    total_possible_rdms += len(self.rois)
                    session_data = self.get_session_data(subject, session)
                    for roi in session_data['available_rdms']:
                        rdm_coverage[roi] += 1
                        
        print(f"  Total possible RDMs: {total_possible_rdms}")
        for roi, count in sorted(rdm_coverage.items()):
            print(f"    {roi}: {count} available ({count/total_possible_rdms*len(self.subjects)*15*100:.1f}% coverage)")


# Test the enhanced parser
if __name__ == "__main__":
    metadata = BOLD5000MetadataEnhanced('data/BOLD5000', verbose=True)
    metadata.print_summary_report()
    
    # Test getting data for a specific session
    print("\n" + "="*60)
    print("Testing Session Data Retrieval")
    print("="*60)
    
    session_data = metadata.get_session_data('CSI1', 1)
    print(f"\nCSI1 Session 1:")
    print(f"  Total trials: {session_data['num_trials']}")
    print(f"  Available images: {session_data['num_available_images']}")
    print(f"  Missing images: {len(session_data['missing_images'])}")
    print(f"  Available RDMs: {len(session_data['available_rdms'])} ROIs")
    
    # Show a sample image path to verify matching
    if session_data['images']:
        sample = session_data['images'][0]
        print(f"\nSample image mapping:")
        print(f"  Presentation name: {sample['presentation_name']}")
        print(f"  Actual file: {sample['image_path'].name}")