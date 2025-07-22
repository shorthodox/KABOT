import sys
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib
import numpy as np
from collections import Counter
import traceback
from ta.volatility import BollingerBands  # Added for Bollinger Bands

# Debugging setup
print("\n=== Script Initialization ===")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Script location: {__file__}")

# Add project root to Python path
try:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))
    print(f"\nAdded to Python path: {project_root}")
except Exception as e:
    print(f"\nError setting Python path: {e}")
    raise

# Import technical indicators with better error handling
try:
    from src.analyzer.technical import TechnicalIndicators
    print("Successfully imported TechnicalIndicators")
except ImportError as e:
    print(f"\nFailed to import TechnicalIndicators: {e}")
    print("Current Python path:")
    print(sys.path)
    print("\nCheck that:")
    print("1. src/analyzer/technical.py exists")
    print("2. It contains a TechnicalIndicators class")
    print("3. There are __init__.py files in src/ and src/analyzer/")
    raise

def prepare_features(df, lookback=5):
    """Enhanced feature engineering with Bollinger Bands Width"""
    try:
        # Validate input
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        required_cols = {'close_price', 'volume'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Create features
        print("\nCreating price features...")
        df['returns'] = df['close_price'].pct_change()
        df['volatility'] = df['returns'].rolling(5).std()
        df['price_change_5'] = df['close_price'].pct_change(5)
        
        # Add Bollinger Bands Width
        print("Adding Bollinger Bands indicators...")
        bb = BollingerBands(close=df['close_price'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        print("Creating volume features...")
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        
        print(f"Creating lag features (lookback={lookback})...")
        for lag in range(1, lookback+1):
            df[f'close_lag_{lag}'] = df['close_price'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'bb_width_lag_{lag}'] = df['bb_width'].shift(lag)  # Lagged BB Width
        
        print("Creating rolling features...")
        windows = [5, 10, 20]
        for window in windows:
            df[f'ma_{window}'] = df['close_price'].rolling(window).mean()
            df[f'std_{window}'] = df['close_price'].rolling(window).std()
            df[f'min_{window}'] = df['close_price'].rolling(window).min()
            df[f'max_{window}'] = df['close_price'].rolling(window).max()
            df[f'bb_width_ma_{window}'] = df['bb_width'].rolling(window).mean()  # Moving average of BB Width
        
        # Clean up
        initial_rows = len(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Handle infinities first
        df.dropna(inplace=True)
        final_rows = len(df)
        print(f"\nDropped {initial_rows - final_rows} rows with NA/Inf values")
        print(f"Final dataset shape: {df.shape}")
        
        return df
    except Exception as e:
        print(f"\nError in prepare_features: {e}")
        raise

def walk_forward_train(df, n_splits=3):
    """Robust time-series walk-forward validation with proper index handling"""
    try:
        # Validate input
        if 'label' not in df.columns:
            raise ValueError("DataFrame must contain 'label' column")

        print("\nPreparing for walk-forward training...")
        
        # Reset index to ensure continuous integer indexing
        df = df.reset_index(drop=True)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        features = [col for col in df.columns if col not in ['label', 'stock', 'date']]
        print(f"Using {len(features)} features for training")

        # Encode labels and convert to numpy array
        le = LabelEncoder()
        y = le.fit_transform(df['label'])
        X = df[features]

        # Data quality checks
        print("\nPerforming data quality checks...")
        
        # 1. Check for infinite values
        with pd.option_context('mode.use_inf_as_na', True):
            inf_mask = np.isinf(X.select_dtypes(include=np.number))
            if inf_mask.any().any():
                print(f"Found {inf_mask.sum().sum()} infinite values - replacing with NA")
                X = X.replace([np.inf, -np.inf], np.nan)

            # 2. Check for extreme values
            extreme_threshold = 1e10
            extreme_mask = (X.select_dtypes(include=np.number).abs() > extreme_threshold)
            if extreme_mask.any().any():
                print(f"Found {extreme_mask.sum().sum()} extreme values - replacing with NA")
                X[X.select_dtypes(include=np.number).abs() > extreme_threshold] = np.nan

            # 3. Drop NA values and maintain alignment
            initial_rows = len(X)
            valid_mask = ~X.isna().any(axis=1)
            X = X[valid_mask]
            y = y[valid_mask.values]  # Convert pandas Series to numpy array for indexing
            print(f"Dropped {initial_rows - len(X)} rows with invalid values")

        # Verify data shape
        if len(X) == 0:
            raise ValueError("No valid data remaining after cleaning")

        # Training setup
        models = []
        reports = []
        feature_importances = []

        print(f"\nStarting {n_splits}-fold time series cross-validation")
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"\n=== Fold {fold + 1}/{n_splits} ===")
            print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
            
            # Ensure indices are within bounds
            train_idx = train_idx[train_idx < len(X)]
            test_idx = test_idx[test_idx < len(X)]
            
            # Prepare data - convert to numpy arrays
            X_train, X_test = X.iloc[train_idx].values, X.iloc[test_idx].values
            y_train, y_test = y[train_idx], y[test_idx]

            # Scale features with validation
            scaler = StandardScaler()
            try:
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Additional check for problematic values after scaling
                if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
                    raise ValueError("NaN or Inf found after scaling")
            except ValueError as e:
                print("\nScaling Error Details:")
                print(f"- NaN in X_train: {np.isnan(X_train).sum()}")
                print(f"- Inf in X_train: {np.isinf(X_train).sum()}")
                print(f"- X_train shape: {X_train.shape}")
                print("First 5 rows of problematic features:")
                problematic_cols = np.where(np.isnan(X_train).any(axis=0))[0]
                print(X.iloc[train_idx].iloc[:, problematic_cols].head())
                raise

            # Handle class imbalance
            class_counts = np.bincount(y_train)
            print(f"Class distribution: {dict(zip(le.classes_, class_counts))}")
            
            # Safely calculate weight ratio
            min_count = max(1, min(class_counts))
            weight_ratio = max(class_counts) / min_count
            print(f"Class weight ratio: {weight_ratio:.2f}")

            # Configure and train model
            model = XGBClassifier(
                objective='multi:softprob',
                eval_metric='mlogloss',
                random_state=42,
                n_estimators=500,
                max_depth=6,
                learning_rate=0.01,
                reg_alpha=0.1,
                reg_lambda=0.1,
                scale_pos_weight=weight_ratio,
                early_stopping_rounds=20,
                subsample=0.8,
                colsample_bytree=0.8
            )

            print("\nTraining model...")
            eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
            model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=10)

            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
            reports.append(report)

            # Safely print metrics
            print("\nFold metrics:")
            print(f"Accuracy: {report.get('accuracy', float('nan')):.4f}")
            weighted_avg = report.get('weighted avg', {})
            print(f"Precision: {weighted_avg.get('precision', float('nan')):.4f}")
            print(f"Recall: {weighted_avg.get('recall', float('nan')):.4f}")
            print(f"F1-score: {weighted_avg.get('f1-score', float('nan')):.4f}")

            # Track feature importance
            fold_importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_,
                'fold': fold
            })
            feature_importances.append(fold_importance)
            models.append((model, scaler))

        # Analyze results
        importance_df = pd.concat(feature_importances)
        avg_importance = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)

        print("\n=== Training Summary ===")
        print("\nTop 20 Features:")
        print(avg_importance.head(20))

        # Select best model
        best_idx = np.argmax([r.get('accuracy', 0) for r in reports])
        best_model, best_scaler = models[best_idx]
        print(f"\nSelected best model from fold {best_idx + 1}")
        print(f"Validation Accuracy: {reports[best_idx].get('accuracy', float('nan')):.4f}")

        return best_model, best_scaler, le, features, avg_importance

    except Exception as e:
        print(f"\nError in walk_forward_train: {e}")
        print(f"Current data shape: {X.shape if 'X' in locals() else 'N/A'}")
        if 'y' in locals():
            y_len = len(np.asarray(y))
        else:
            y_len = 'N/A'
        print(f"Label length: {y_len}")
        raise

def train_model(data_path='data/processed/trading_dataset.csv', 
               model_output_path='models/xgb_model_final.pkl'):
    """Modified training pipeline with Bollinger Bands support"""
    try:
        print("\n=== Starting Training Pipeline ===")
        
        # Validate paths
        data_path = Path(data_path)
        print(f"\nLooking for data at: {data_path.resolve()}")
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        # Load data
        print("\nLoading data...")
        raw_df = pd.read_csv(data_path)
        print(f"Raw data shape: {raw_df.shape}")
        print("\nOriginal columns:")
        print(raw_df.columns.tolist())
        
        # Verify minimum required columns
        required_columns = {'close_price', 'volume', 'label', 'stock'}
        missing = required_columns - set(raw_df.columns)
        if missing:
            raise ValueError(f"Missing absolutely required columns: {missing}")
        
        # Add dummy date column if missing (using index as proxy)
        if 'date' not in raw_df.columns:
            print("\nNo date column found - using index as time reference")
            raw_df['date'] = pd.to_datetime(raw_df.index)
        
        # Create synthetic OHLC if missing
        ohlc_cols = {'open_price', 'high_price', 'low_price'}
        missing_ohlc = ohlc_cols - set(raw_df.columns)
        if missing_ohlc:
            print(f"\nCreating synthetic OHLC data for missing columns: {missing_ohlc}")
            if 'close_price' in raw_df.columns:
                raw_df['open_price'] = raw_df['close_price'].shift(1)
                raw_df['high_price'] = raw_df[['open_price', 'close_price']].max(axis=1)
                raw_df['low_price'] = raw_df[['open_price', 'close_price']].min(axis=1)
                raw_df.dropna(subset=['open_price', 'high_price', 'low_price'], inplace=True)
        
        print("\nProcessed data columns:")
        print(raw_df.columns.tolist())
        
        # Feature engineering
        print("\nEngineering features...")
        feature_df = prepare_features(raw_df)
        
        print("\nFinal dataset info:")
        print(feature_df.info())
        print("\nLabel distribution:")
        print(feature_df['label'].value_counts(normalize=True))
        
        # Model training
        print("\nStarting model training...")
        model, scaler, le, features, importance = walk_forward_train(feature_df)
        
        # Save model
        print("\nSaving model artifacts...")
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        artifacts = {
            'model': model,
            'scaler': scaler,
            'label_encoder': le,
            'features': features,
            'feature_importance': importance,
            'class_distribution': dict(Counter(feature_df['label'])),
            'note': 'Trained with Bollinger Bands Width feature'
        }
        joblib.dump(artifacts, model_output_path)
        print(f"\nModel saved to {model_output_path}")
        
        return model_output_path
        
    except Exception as e:
        print(f"\nError in train_model: {e}")
        print("\nTraceback:")
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    try:
        print("\n=== Script Execution Started ===")
        model_path = train_model()
        print(f"\n=== Training Completed Successfully ===")
        print(f"Model saved to: {model_path}")
    except Exception as e:
        print(f"\n!!! Script Failed: {e}")
        sys.exit(1)