import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import os

class EEGExplorer:
    """
    A class to explore EEG data from the dataset focused on AD analysis.
    """
    
    def __init__(self, data_path: str = "data/PLOSONE2020_DATA_v1.1.csv"):
        """Initialize the EEG Explorer with data path."""
        self.data = self._load_data(data_path)
        self.results_path = self._create_results_directory()
        self.frequency_bands = {
            'Delta': (1, 3),
            'Theta': (3, 7),
            'Alpha': (8, 13),
            'Beta': (13, 30)
        }
        self.regions = ['Frontal', 'Central', 'Parietal', 'Occipital', 'Temporal']
        
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load EEG data from CSV file and perform initial validation."""
        try:
            data = pd.read_csv(data_path)
            required_columns = ['Condition', 'YY_EEG']
            if not all(col in data.columns for col in required_columns):
                raise ValueError("Missing required columns in data")
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {data_path}")
            
    def _create_results_directory(self) -> Path:
        """Create timestamped directory for results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = Path('results') / timestamp
        os.makedirs(results_dir / 'plots', exist_ok=True)
        os.makedirs(results_dir / 'stats', exist_ok=True)
        return results_dir

    def _get_band_columns(self, band: str) -> list:
        """Get columns related to a specific frequency band."""
        band_cols = [col for col in self.data.columns if band in col and '_Rel' not in col]
        return band_cols

    def _save_plot(self, plt, name: str):
        """Save plot to results directory."""
        plt.savefig(self.results_path / 'plots' / f'{name}.png')
        plt.close()

    def plot_band_distributions(self, band: str):
        """Create distribution plots for specified frequency band."""
        band_cols = self._get_band_columns(band)
        
        plt.figure(figsize=(15, 10))
        for col in band_cols[:5]:  # Plot first 5 channels as example
            sns.kdeplot(data=self.data, x=col, hue='Condition')
        plt.title(f'{band} Band Power Distribution by Condition')
        plt.xlabel('Power')
        plt.ylabel('Density')
        self._save_plot(plt, f'{band}_distribution')

    def plot_regional_comparisons(self):
        """Create plots comparing power across brain regions between groups."""
        for region in self.regions:
            regional_cols = [col for col in self.data.columns if region in col and 'Alpha' in col]
            if not regional_cols:
                continue

            plt.figure(figsize=(12, 8))
            data_to_plot = self.data.melt(
                id_vars=['Condition'], 
                value_vars=regional_cols,
                var_name='Channel',
                value_name='Power'
            )
            
            sns.boxplot(data=data_to_plot, x='Channel', y='Power', hue='Condition')
            plt.title(f'{region} Region Power Comparison')
            plt.xticks(rotation=45)
            self._save_plot(plt, f'{region}_comparison')

    def plot_TAR_analysis(self):
        """Create visualizations for Theta/Alpha ratio analysis."""
        tar_cols = [col for col in self.data.columns if 'TAR' in col]
        
        plt.figure(figsize=(15, 10))
        for col in tar_cols[:5]:  # Plot first 5 channels
            sns.boxplot(data=self.data, x='Condition', y=col)
            plt.title(f'Theta/Alpha Ratio by Condition - {col}')
            self._save_plot(plt, f'TAR_{col}')

    def plot_coherence_patterns(self):
        """Analyze and visualize coherence patterns."""
        # Assuming coherence metrics are in the dataset
        coherence_cols = [col for col in self.data.columns if any(
            f'{region}-' in col for region in self.regions)]
        
        if coherence_cols:
            plt.figure(figsize=(15, 10))
            sns.heatmap(
                self.data[coherence_cols].corr(), 
                cmap='coolwarm',
                center=0
            )
            plt.title('Coherence Pattern Correlation Matrix')
            self._save_plot(plt, 'coherence_patterns')

    def generate_summary_stats(self):
        """Generate summary statistics for all analyses."""
        stats = {}
        
        # Basic group statistics
        stats['group_counts'] = self.data['Condition'].value_counts().to_dict()
        
        # Band power statistics
        for band in self.frequency_bands.keys():
            band_cols = self._get_band_columns(band)
            stats[f'{band}_stats'] = self.data[band_cols].describe().to_dict()
        
        # Save statistics
        pd.DataFrame(stats).to_csv(self.results_path / 'stats' / 'summary_stats.csv')

    def run_full_analysis(self):
        """Run complete analysis pipeline and save all results."""
        # Run all analyses
        for band in self.frequency_bands.keys():
            self.plot_band_distributions(band)
        
        self.plot_regional_comparisons()
        self.plot_TAR_analysis()
        self.plot_coherence_patterns()
        self.generate_summary_stats()

if __name__ == "__main__":
    explorer = EEGExplorer()
    explorer.run_full_analysis()