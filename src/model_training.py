import pandas as pd
import numpy as np
import optuna
from pathlib import Path
import joblib
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from imblearn.over_sampling import SMOTE

# ML imports
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, 
    cross_validate
)
from sklearn.metrics import (
    roc_auc_score, classification_report, 
    f1_score, confusion_matrix, balanced_accuracy_score
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

class ModelTrainer:
    """Class for training and optimizing ML models for EEG classification."""
    
    def __init__(
        self,
        X_path: str = "data/X_ml.csv",
        y_path: str = "data/y_ml.csv",
        random_state: int = 42
    ):
        self.X_path = X_path
        self.y_path = y_path
        self.random_state = random_state
        self.X = None
        self.y = None
        self.best_model = None
        self.best_params = None
        self.results_path = self._setup_results_dir()
        
        # Define model types
        self.models = {
            'xgb': XGBClassifier,
            'catboost': CatBoostClassifier,
            'gboost': GradientBoostingClassifier,
            'rf': RandomForestClassifier,
            # 'svc': SVC
        }
        
    def _setup_results_dir(self) -> Path:
        """Create timestamped results directory."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = Path('results') / timestamp
        for subdir in ['models', 'plots', 'metrics']:
            (results_dir / subdir).mkdir(parents=True, exist_ok=True)
        return results_dir

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare the dataset."""
        self.X = pd.read_csv(self.X_path)
        self.y = pd.read_csv(self.y_path)['Condition']
        
        # Filter for AD vs HC3 only
        mask = self.y.isin(['AD', 'HC3'])
        self.X = self.X[mask].reset_index(drop=True)
        self.y = self.y[mask].reset_index(drop=True)
        
        # Encode labels: AD -> 0, HC3 -> 1
        self.y = (self.y == 'HC3').astype(int)
        
        # Apply SMOTE for balanced sampling
        smote = SMOTE(random_state=self.random_state)
        self.X, self.y = smote.fit_resample(self.X, self.y)
        
        return self.X, self.y

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for model selection and optimization."""
        # Select model type
        model_type = trial.suggest_categorical('model_type', list(self.models.keys()))
        
        # Define model-specific parameters
        if model_type == 'xgb':
            params = {
                'model__n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'model__max_depth': trial.suggest_int('max_depth', 3, 10),
                'model__learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                'model__subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'model__colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'model__min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'model__gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True)
            }
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', XGBClassifier(random_state=self.random_state))
            ])
            
        elif model_type == 'catboost':
            params = {
                'model__iterations': trial.suggest_int('iterations', 100, 1000),
                'model__depth': trial.suggest_int('depth', 3, 10),
                'model__learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                'model__l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True)
            }
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', CatBoostClassifier(random_state=self.random_state))
            ])
            
        elif model_type == 'rf':
            params = {
                'model__n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'model__max_depth': trial.suggest_int('max_depth', 3, 15),
                'model__min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'model__min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(random_state=self.random_state))
            ])

        elif model_type == 'svc':
            params = {
                'model__C': trial.suggest_float('C', 1e-3, 100, log=True),
                'model__gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
                'model__kernel': trial.suggest_categorical('kernel', ['rbf', 'linear'])
            }
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVC(random_state=self.random_state, probability=True))
            ])

        else:  # gboost
            params = {
                'model__n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'model__max_depth': trial.suggest_int('max_depth', 3, 10),
                'model__learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                'model__subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'model__min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'model__min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingClassifier(random_state=self.random_state))
            ])

        model.set_params(**params)
        
        # Use stratified CV with balanced accuracy
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(
            model, self.X, self.y, 
            scoring='balanced_accuracy',
            cv=cv, 
            n_jobs=-1
        )
        
        return scores.mean()

    def optimize_and_train(self, n_trials: int = 100) -> Any:
        """Optimize model selection and hyperparameters."""
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        model_type = self.best_params.pop('model_type')  # Remove model_type from params
        model_class = self.models[model_type]
        
        # Create and fit the best model
        self.best_model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model_class(random_state=self.random_state, **self.best_params))
        ])
        self.best_model.fit(self.X, self.y)
        
        return self.best_model

    def run_pipeline(self, n_trials: int = 100):
        """Run complete training pipeline."""
        # Load data
        self.load_data()
        
        # Train best model
        model = self.optimize_and_train(n_trials)
            
        # Evaluate
        metrics = self.evaluate_model(model)
        
        # Save results
        self.save_results(model, metrics)
        
        return model, metrics

    def evaluate_model(self, model) -> Dict[str, Any]:
        """Evaluate model performance with proper scaling."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Use cross_validate to get predictions while respecting the pipeline
        cv_results = cross_validate(
            model, 
            self.X, 
            self.y, 
            cv=cv,
            scoring={
                'balanced_accuracy': 'balanced_accuracy',
                'roc_auc': 'roc_auc',
                'f1': 'f1_weighted'
            },
            return_estimator=True,
            return_train_score=False
        )
        
        # Get predictions using the fitted pipelines
        y_pred = np.zeros_like(self.y)
        y_prob = np.zeros_like(self.y, dtype=float)
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(self.X, self.y)):
            fitted_pipeline = cv_results['estimator'][fold_idx]
            y_pred[test_idx] = fitted_pipeline.predict(self.X.iloc[test_idx])
            y_prob[test_idx] = fitted_pipeline.predict_proba(self.X.iloc[test_idx])[:, 1]
        
        # Compute metrics
        metrics = {
            'balanced_accuracy': balanced_accuracy_score(self.y, y_pred),
            'roc_auc': roc_auc_score(self.y, y_prob),
            'f1': f1_score(self.y, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(self.y, y_pred),
            'classification_report': classification_report(self.y, y_pred)
        }
        
        return metrics

    def save_results(self, model, metrics: Dict[str, Any]):
        """Save model and results."""
        # Save model
        model_filename = f'best_model_{type(model).__name__}.joblib'
        joblib.dump(model, self.results_path / 'models' / model_filename)
        
        # Save metrics (excluding confusion matrix)
        metrics_for_csv = {
            k: v for k, v in metrics.items() 
            if k not in ['confusion_matrix']
        }
        metrics_df = pd.DataFrame([metrics_for_csv])
        metrics_df.to_csv(self.results_path / 'metrics' / 'metrics.csv')
        
        # Save model metadata
        model_metadata = {
            'model_name': type(model).__name__,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': model.get_params(),
            **metrics_for_csv  # Include all metrics
        }
        
        # Convert to DataFrame and save
        metadata_df = pd.DataFrame([model_metadata])
        metadata_df.to_csv(self.results_path / 'metrics' / 'model_metadata.csv')
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            metrics['confusion_matrix'], 
            annot=True, 
            fmt='d',
            xticklabels=['AD', 'HC3'],
            yticklabels=['AD', 'HC3']
        )
        plt.title('Confusion Matrix')
        plt.savefig(self.results_path / 'plots' / 'confusion_matrix.png')
        plt.close()

if __name__ == "__main__":
    trainer = ModelTrainer()
    model, metrics = trainer.run_pipeline()