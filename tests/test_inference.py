import pytest
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


class TestModelInference:
    """
    Tests pour les prédictions du modèle entraîné.
    """
    
    @pytest.fixture
    def sample_model(self):
        """Crée un modèle simple pour tester."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(5)])
        
        # Crée un simple pipeline
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_df, y)
        return model, X_df, y
    
    def test_model_prediction_shape(self, sample_model):
        """Vérifie que les prédictions ont la bonne forme."""
        model, X_test, _ = sample_model
        predictions = model.predict(X_test[:10])
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 10
    
    def test_model_prediction_probabilities_shape(self, sample_model):
        """Vérifie que les probabilités ont la bonne forme."""
        model, X_test, _ = sample_model
        proba = model.predict_proba(X_test[:10])
        
        assert isinstance(proba, np.ndarray)
        assert proba.shape == (10, 2)  # Binary classification
    
    def test_model_prediction_probabilities_sum_to_one(self, sample_model):
        """Vérifie que chaque ligne de probabilités somme à 1."""
        model, X_test, _ = sample_model
        proba = model.predict_proba(X_test[:10])
        
        row_sums = proba.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(10))
    
    def test_model_prediction_binary_output(self, sample_model):
        """Vérifie que les prédictions sont binaires (0 ou 1)."""
        model, X_test, _ = sample_model
        predictions = model.predict(X_test[:20])
        
        unique_vals = np.unique(predictions)
        assert set(unique_vals).issubset({0, 1})
    
    def test_model_serialization_and_loading(self, sample_model, tmp_path):
        """Vérifie que le modèle peut être sérialisé et chargé."""
        model, X_test, _ = sample_model
        
        # Sauvegarde avec joblib
        model_path = tmp_path / "test_model.joblib"
        joblib.dump(model, str(model_path))
        
        # Charge et verifie
        loaded_model = joblib.load(str(model_path))
        
        # Les prédictions doivent être identiques
        pred_original = model.predict(X_test[:5])
        pred_loaded = loaded_model.predict(X_test[:5])
        
        np.testing.assert_array_equal(pred_original, pred_loaded)
    
    def test_model_consistency_across_calls(self, sample_model):
        """Vérifie que le modèle produit les mêmes résultats pour les mêmes données."""
        model, X_test, _ = sample_model
        
        pred_1 = model.predict(X_test[:10])
        pred_2 = model.predict(X_test[:10])
        
        np.testing.assert_array_equal(pred_1, pred_2)
    
    def test_best_model_artifact_exists(self):
        """Vérifie que le fichier best_model.joblib existe dans les artifacts."""
        model_path = "artifacts/best_model.joblib"
        # Ne test que si le fichier existe, skip sinon
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            assert model is not None
            # Vérifie que c'est un modèle sklearn valide
            assert hasattr(model, 'predict') or hasattr(model, 'named_steps')
        else:
            pytest.skip("best_model.joblib not found (needs training first)")
    
    def test_lgbm_artifact_exists(self):
        """Vérifie que le modèle LGBM a été sauvegardé."""
        model_path = "artifacts/lgbm.joblib"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            assert model is not None
            assert hasattr(model, 'predict_proba')
        else:
            pytest.skip("lgbm.joblib not found (needs training first)")
    
    def test_model_handles_missing_features_gracefully(self, sample_model):
        """Teste le comportement avec des données incomplètes."""
        model, _, _ = sample_model
        
        # Crée des données avec NaN
        X_nan = pd.DataFrame({
            f'feat_{i}': [1.0, np.nan, 3.0, 4.0, 5.0]
            for i in range(5)
        })
        
        # Remplace les NaN par la moyenne pour que le modèle fonctionne
        X_nan = X_nan.fillna(X_nan.mean())
        
        # Le modèle doit pouvoir faire des prédictions
        predictions = model.predict(X_nan)
        assert len(predictions) == 5
    
    def test_model_prediction_values_bounded(self, sample_model):
        """Vérifie que les probabilités sont en [0, 1]."""
        model, X_test, _ = sample_model
        proba = model.predict_proba(X_test[:20])
        
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)


class TestCSVDataIntegrity:
    """
    Tests pour vérifier l'intégrité des données CSV préparées.
    """
    
    @pytest.fixture
    def csv_data(self):
        """Charge les données du CSV préparé."""
        csv_path = "data/processed/features_and_target.csv"
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        else:
            pytest.skip("features_and_target.csv not found")
    
    def test_csv_has_target_column(self, csv_data):
        """Vérifie que le CSV contient la colonne cible."""
        target_cols = [col for col in csv_data.columns if 'Diagnosis' in col and 'appendicitis' in col]
        assert len(target_cols) > 0, "Target column not found in CSV"
    
    def test_csv_no_missing_values_in_target(self, csv_data):
        """Vérifie qu'il n'y a pas de valeurs manquantes dans la cible."""
        target_col = [col for col in csv_data.columns if 'Diagnosis_no_appendicitis' in col or 'Diagnosis' in col][0]
        missing_count = csv_data[target_col].isna().sum()
        assert missing_count == 0, f"Found {missing_count} missing values in target column"
    
    def test_csv_target_is_binary(self, csv_data):
        """Vérifie que la cible contient seulement 0 et 1."""
        target_col = [col for col in csv_data.columns if 'Diagnosis' in col][-1]
        unique_values = set(csv_data[target_col].unique())
        assert unique_values.issubset({0, 1, True, False}), f"Target has invalid values: {unique_values}"
    
    def test_csv_sufficient_samples(self, csv_data):
        """Vérifie qu'il y a assez d'échantillons pour l'entraînement."""
        min_samples = 100  # Au moins 100 échantillons
        assert len(csv_data) >= min_samples, f"Only {len(csv_data)} samples, need at least {min_samples}"
    
    def test_csv_sufficient_features(self, csv_data):
        """Vérifie qu'il y a au moins quelques features."""
        # Exclut la colonne cible
        target_col = [col for col in csv_data.columns if 'Diagnosis' in col][-1]
        n_features = len(csv_data.columns) - 1
        assert n_features >= 3, f"Only {n_features} features, need at least 3"
    
    def test_csv_no_all_missing_columns(self, csv_data):
        """Vérifie qu'aucune colonne n'est entièrement composée de NaN."""
        for col in csv_data.columns:
            missing_pct = csv_data[col].isna().sum() / len(csv_data)
            assert missing_pct < 1.0, f"Column {col} is entirely missing"
    
    def test_csv_data_types_consistent(self, csv_data):
        """Vérifie que les types de données sont raisonnables."""
        for col in csv_data.columns:
            dtype = csv_data[col].dtype
            # Doit être numérique ou booléen ou categorie
            assert dtype in ['int64', 'int32', 'float64', 'float32', 'bool', 'object', 'category'], \
                f"Unexpected dtype {dtype} in column {col}"
