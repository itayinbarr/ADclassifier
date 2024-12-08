import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, List

class EEGPreprocessor:
    """
    Class for preprocessing EEG data focusing on frequency band features.
    Extracts frequency information from raw EEG data for AD analysis.
    """
    
    def __init__(self, data_path: str = "data/PLOSONE2020_DATA_v1.1.csv"):
        """Initialize the preprocessor with data path."""
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        self.setup_logging()
        
        # Define frequency bands of interest
        self.frequency_bands = {
            'Delta': '_1_3',          # 1-3 Hz
            'ThetaSlow': '3_5',       # 3-5 Hz
            'ThetaFast': '5_7',       # 5-7 Hz
            'Theta': '3_7',           # 3-7 Hz (overall theta)
            'AlphaSlow': '8_10',      # 8-10 Hz
            'AlphaFast': '10_13',     # 10-13 Hz
            'Alpha': '8_13',          # 8-13 Hz (overall alpha)
            'BetaSlow': '13_20',      # 13-20 Hz
            'BetaFast': '21_30',      # 21-30 Hz
            'Beta': '13_30'           # 13-30 Hz (overall beta)
        }
        
        # Define brain regions
        self.regions = {
            'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz'],
            'Central': ['C3', 'C4', 'Cz'],
            'Parietal': ['P3', 'P4', 'Pz', 'POz'],
            'Occipital': ['O1', 'O2'],
            'Temporal': ['T3', 'T4', 'T5', 'T6']
        }
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_and_clean(self) -> pd.DataFrame:
        """Load and clean the EEG data."""
        self.logger.info("Loading data...")
        self.data = pd.read_csv(self.data_path)
        
        # Remove rows with invalid data
        self.data = self.data[self.data['Valid_Data'] == True]
        
        # Select only relative power columns (_Rel) and condition
        power_cols = [col for col in self.data.columns if any(
            band in col for band in self.frequency_bands.values()) and '_Rel' in col]
        
        self.data = self.data[power_cols + ['Condition']]
        
        self.logger.info(f"Data loaded and cleaned. Shape: {self.data.shape}")
        return self.data

    def extract_band_features(self) -> pd.DataFrame:
        """
        Extract frequency band features by region.
        Returns DataFrame with regional frequency band powers.
        """
        features = pd.DataFrame()
        
        # For each frequency band
        for band_name, band_suffix in self.frequency_bands.items():
            # For each brain region
            for region_name, channels in self.regions.items():
                # Get columns for this band and region
                cols = [col for col in self.data.columns 
                       if band_suffix in col 
                       and '_Rel' in col 
                       and any(ch in col for ch in channels)]
                
                if cols:
                    # Calculate mean power for this band in this region
                    features[f'{band_name}_{region_name}_mean'] = self.data[cols].mean(axis=1)
                    
            # Calculate overall mean for this band
            band_cols = [col for col in self.data.columns 
                        if band_suffix in col and '_Rel' in col]
            features[f'{band_name}_mean'] = self.data[band_cols].mean(axis=1)
        
        return features

    def create_processed_dataset(self) -> pd.DataFrame:
        """Create the final processed dataset for modeling."""
        self.logger.info("Creating processed dataset...")
        
        # Load and clean data if not already done
        if self.data is None:
            self.load_and_clean()
            
        # Extract frequency band features
        band_features = self.extract_band_features()
        
        # Combine features with condition
        self.processed_data = pd.concat(
            [band_features, self.data[['Condition']]],
            axis=1
        )
        
        self.logger.info(f"Processed dataset created. Shape: {self.processed_data.shape}")
        return self.processed_data

    def save_processed_data(self, output_path: str = None):
        """Save processed dataset to file."""
        if output_path is None:
            output_path = 'data/processed_data.csv'
            
        if self.processed_data is None:
            self.create_processed_dataset()
            
        self.processed_data.to_csv(output_path, index=False)
        self.logger.info(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    preprocessor = EEGPreprocessor()
    preprocessor.create_processed_dataset()
    preprocessor.save_processed_data()