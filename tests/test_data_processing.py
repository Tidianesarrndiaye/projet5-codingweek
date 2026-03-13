import pytest
import pandas as pd
import numpy as np
from src.data_processing import (
    handle_missing_values,
    handle_outliers,
    encode_features,
    drop_high_missing_columns,
    optimize_memory,
    preprocess_pipeline
)


class TestHandleMissingValues:
    """
    Tests pour la gestion des valeurs manquantes.
    """
    
    def test_handle_missing_values_numeric_median_imputation(self):
        """Vérifie que les colonnes numériques sont remplies par la médiane."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0],
            'B': [10, 20, 30, 40]
        })
        result = handle_missing_values(df)
        assert result['A'].isna().sum() == 0
        assert result['A'].iloc[2] == 2.0  # médiane de [1, 2, 4]
    
    def test_handle_missing_values_categorical_mode_imputation(self):
        """Vérifie que les colonnes catégorielles sont remplies par le mode."""
        df = pd.DataFrame({
            'cat_col': ['A', 'A', None, 'B', 'A']
        })
        result = handle_missing_values(df)
        assert result['cat_col'].isna().sum() == 0
    
    def test_handle_missing_values_no_missing(self):
        """Vérifie que les dataframes sans valeur manquante restent inchangés."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        result = handle_missing_values(df)
        pd.testing.assert_frame_equal(result, df)


class TestHandleOutliers:
    """
    Tests pour la détection et le traitement des outliers (IQR clipping).
    """
    
    def test_handle_outliers_clips_extreme_values(self):
        """Vérifie que les valeurs en dehors de [Q1-1.5*IQR, Q3+1.5*IQR] sont clippées."""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 est un outlier
        })
        result = handle_outliers(df, iqr_factor=1.5)
        # Le max ne doit pas être 100 après clipping
        assert result['values'].max() < 100
    
    def test_handle_outliers_preserves_inliers(self):
        """Vérifie que les valeurs normales ne sont pas modifiées."""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 6, 7, 8, 9]
        })
        result = handle_outliers(df)
        pd.testing.assert_frame_equal(result, df)
    
    def test_handle_outliers_no_numeric_columns(self):
        """Vérifie que les colonnes non-numériques sont ignorées."""
        df = pd.DataFrame({
            'cat': ['a', 'b', 'c']
        })
        result = handle_outliers(df)
        pd.testing.assert_frame_equal(result, df)


class TestEncodeFeatures:
    """
    Tests pour l'encodage one-hot des colonnes catégorielles.
    """
    
    def test_encode_features_creates_dummy_variables(self):
        """Vérifie que le one-hot encoding crée les colonnes appropriées."""
        df = pd.DataFrame({
            'cat': ['A', 'B', 'A', 'C'],
            'num': [1, 2, 3, 4]
        })
        encoded, meta = encode_features(df, drop_first=True)
        # Vérifie que des colonnes one-hot ont été créées
        assert 'cat_B' in encoded.columns or 'cat_C' in encoded.columns
        assert 'num' in encoded.columns
    
    def test_encode_features_all_numeric(self):
        """Vérifie que les dataframes tout numériques restent inchangés."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        encoded, meta = encode_features(df)
        assert len(encoded.columns) == 2
        assert list(encoded.columns) == ['A', 'B']
    
    def test_encode_features_returns_meta(self):
        """Vérifie que les métadonnées d'encodage sont retournées."""
        df = pd.DataFrame({
            'cat': ['X', 'Y'],
            'num': [1, 2]
        })
        encoded, meta = encode_features(df)
        assert 'encoded_columns' in meta
        assert 'categorical_columns' in meta


class TestDropHighMissingColumns:
    """
    Tests pour la suppression des colonnes avec trop de valeurs manquantes.
    """
    
    def test_drop_high_missing_columns_removes_sparse_columns(self):
        """Vérifie que les colonnes avec >threshold % de NaN sont supprimées."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [np.nan, np.nan, np.nan]  # 100% NaN
        })
        result = drop_high_missing_columns(df, threshold=0.05)
        assert 'B' not in result.columns
        assert 'A' in result.columns
    
    def test_drop_high_missing_columns_preserves_dense_columns(self):
        """Vérifie que les colonnes avec <threshold % de NaN sont gardées."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan],
            'B': [4, 5, 6]
        })
        result = drop_high_missing_columns(df, threshold=0.5)
        assert 'A' in result.columns  # Only 33% NaN
        assert 'B' in result.columns


class TestOptimizeMemory:
    """
    Tests pour l'optimisation mémoire des types de colonnes.
    """
    
    def test_optimize_memory_reduces_memory_usage(self):
        """Vérifie que optimize_memory réduit la consommation mémoire."""
        df = pd.DataFrame({
            'int_col': np.array([1, 2, 3, 4, 5], dtype='int64'),
            'float_col': np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype='float64'),
            'cat_col': ['a', 'b', 'c', 'a', 'b']  # Candidate pour category
        })
        mem_before = df.memory_usage(deep=True).sum()
        df_optimized = optimize_memory(df)
        mem_after = df_optimized.memory_usage(deep=True).sum()
        # La mémoire optimisée doit être <= à la mémoire avant
        assert mem_after <= mem_before
    
    def test_optimize_memory_preserves_data_integrity(self):
        """Vérifie que les données ne sont pas modifiées, seulement le type."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [1.5, 2.5, 3.5, 4.5]
        })
        df_orig = df.copy()
        df_optimized = optimize_memory(df)
        # Les valeurs doivent être identiques (même après conversion de type)
        pd.testing.assert_frame_equal(
            df_orig.astype('float64'), 
            df_optimized.astype('float64')
        )


class TestPreprocessPipeline:
    """
    Tests pour le pipeline complet de prétraitement.
    """
    
    def test_preprocess_pipeline_returns_dataframe(self):
        """Vérifie que le pipeline retourne un DataFrame."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': ['a', 'b', None, 'a'],
            'C': [100, 10, 20, 5]
        })
        result, meta = preprocess_pipeline(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # N'a pas supprimé de lignes
    
    def test_preprocess_pipeline_handles_mixed_data(self):
        """Vérifie que le pipeline traite données mixtes (num + cat)."""
        df = pd.DataFrame({
            'numeric': [1.0, 2.0, 3.0, 4.0],
            'categorical': ['X', 'Y', 'Z', 'X']
        })
        result, meta = preprocess_pipeline(df)
        assert result.isna().sum().sum() == 0  # Aucune valeur manquante
        assert len(result) == 4  # Bon nombre de lignes
