import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
import logging
from pathlib import Path
from typing import Tuple

class FeatureEngineer:
    """Class for feature engineering of EEG data with mutual information selection."""
    
    def __init__(self, data_path: str = "data/processed_data.csv"):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> pd.DataFrame:
        """Load the processed data."""
        self.logger.info(f"Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        return self.data
        
    def create_feature_matrix(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Create feature matrix and target vector."""
        if self.data is None:
            self.load_data()
            
        # Separate features and target
        self.y = self.data['Condition']
        self.X = self.data.drop('Condition', axis=1)
        
        # Log number of features
        self.logger.info(f"Initial feature matrix shape: {self.X.shape}")
        
        # Handle missing values
        self.X = self.X.fillna(self.X.mean())
        self.y = self.y[self.X.index]
        
        self.feature_names = self.X.columns.tolist()
        
        return self.X, self.y
    
    def select_features_mi(self, n_features: int = 20) -> pd.DataFrame:
        """
        Select features using mutual information.
        
        Args:
            n_features: Number of features to select
            
        Returns:
            pd.DataFrame: Selected features with their MI scores
        """
        if self.X is None or self.y is None:
            self.create_feature_matrix()
            
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(self.X, self.y, random_state=42)
        
        # Create DataFrame with features and their MI scores
        feature_scores = pd.DataFrame({
            'feature': self.feature_names,
            'mi_score': mi_scores
        })
        
        # Sort by MI score and select top features
        feature_scores = feature_scores.sort_values('mi_score', ascending=False)
        selected_features = feature_scores.head(n_features)
        
        self.logger.info(f"Selected {n_features} features using mutual information")
        self.logger.info("\nTop 5 features:")
        self.logger.info(selected_features.head())
        
        return selected_features
    
    def apply_pca(self, X: pd.DataFrame, n_components: float = 0.99) -> pd.DataFrame:
        """Apply PCA to reduce dimensionality."""
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Log explained variance ratio
        cum_var_ratio = np.cumsum(pca.explained_variance_ratio_)
        n_components_selected = len(cum_var_ratio)
        
        self.logger.info(f"PCA reduced dimensions to {n_components_selected} components")
        self.logger.info(f"Cumulative explained variance ratio: {cum_var_ratio[-1]:.3f}")
        
        # Create column names for PCA features
        pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        return pd.DataFrame(X_pca, columns=pca_columns)
    
    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features using StandardScaler."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)
    
    def create_mi_importance_report(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create a report of feature importances using mutual information.
        
        Args:
            X: Feature matrix
            
        Returns:
            pd.DataFrame: Feature importance report
        """
        if self.y is None:
            self.create_feature_matrix()
            
        mi_scores = mutual_info_classif(X, self.y, random_state=42)
        
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'MI_Score': mi_scores
        })
        
        return importance_df.sort_values('MI_Score', ascending=False)
    
    def save_ml_dataset(self, X: pd.DataFrame, y: pd.Series, output_dir: str = "data"):
        """
        Save the prepared ML dataset.
            
        Args:
            X: Prepared feature matrix
            y: Target vector
            output_dir: Directory to save the files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
            
        # Save features and target
        X.to_csv(output_path / 'X_ml.csv', index=False)
        y.to_csv(output_path / 'y_ml.csv', index=False)
            
        self.logger.info(f"ML dataset saved to {output_path}")

    def select_unique_frequency_features(self, max_features: int = 8) -> pd.DataFrame:
        """
        Select features ensuring unique frequency representations.
        
        Args:
            max_features: Maximum number of unique frequency features to select
            
        Returns:
            pd.DataFrame: Selected features with their MI scores
        """
        if self.X is None or self.y is None:
            self.create_feature_matrix()
            
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(self.X, self.y, random_state=42)
        
        # Create initial feature scores dataframe
        feature_scores = pd.DataFrame({
            'feature': self.feature_names,
            'mi_score': mi_scores
        })
        
        # Define frequency bands
        frequency_bands = {
            'Delta': ['Delta'],
            'Theta': ['Theta'],
            'Alpha': ['Alpha'],
            'BetaSlow': ['BetaSlow'],
            'BetaFast': ['BetaFast'],
            'Beta': ['Beta'],
            'PDDM': ['PDDM']  # Power Distribution Distance Measure
        }
        
        # Function to identify frequency band of a feature
        def get_frequency_band(feature_name: str) -> str:
            for band, keywords in frequency_bands.items():
                if any(keyword in feature_name for keyword in keywords):
                    return band
            return 'Other'
        
        # Add frequency band column
        feature_scores['frequency_band'] = feature_scores['feature'].apply(get_frequency_band)
        
        # Select top feature for each frequency band
        selected_features = []
        
        for band in frequency_bands.keys():
            band_features = feature_scores[feature_scores['frequency_band'] == band]
            if not band_features.empty:
                top_feature = band_features.nlargest(1, 'mi_score')
                selected_features.append(top_feature)
        
        # Combine and sort selected features
        selected_features_df = pd.concat(selected_features)
        selected_features_df = selected_features_df.nlargest(max_features, 'mi_score')
        
        # Log selections
        self.logger.info(f"\nSelected {len(selected_features_df)} unique frequency features:")
        for _, row in selected_features_df.iterrows():
            self.logger.info(f"{row['feature']}: {row['mi_score']:.4f} (Band: {row['frequency_band']})")
        
        return selected_features_df

    def create_ml_ready_dataset(
        self,
        use_pca: bool = True,
        n_components: float = 0.99,
        max_features: int = 8
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create and save a complete ML-ready dataset with unique frequency features.
        """
        # Load and prepare initial feature matrix
        self.create_feature_matrix()
        
        # Select unique frequency features (using unscaled features)
        selected_features = self.select_unique_frequency_features(max_features=max_features)
        X_selected = self.X[selected_features['feature']]
        
        # Apply PCA if requested
        if use_pca:
            X_prepared = self.apply_pca(X_selected, n_components=n_components)
        else:
            X_prepared = X_selected
        
        # Save feature importance report
        selected_features.to_csv('data/feature_importance.csv', index=False)
        
        # Save ML dataset
        self.save_ml_dataset(X_prepared, self.y)
        
        return X_prepared, self.y


    
if __name__ == "__main__":
    engineer = FeatureEngineer()
    X, y = engineer.create_ml_ready_dataset()