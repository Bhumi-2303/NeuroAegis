import numpy as np

from app.services.explainer import normalize_shap_output


def test_normalize_shap_output_list_of_arrays():
    # Typically returned by LGBMClassifier for binary classification
    # List of length 2 (for 2 classes). Each is shape (n_samples, n_features)
    shap_values = [
        np.array([[0.1, 0.2, 0.3]]), # Class 0
        np.array([[-0.1, -0.2, -0.3]]) # Class 1
    ]
    expected_value = [0.5, 0.5] # List of base values
    
    values, base = normalize_shap_output(shap_values, expected_value)
    
    assert base == 0.5
    assert np.allclose(values, [-0.1, -0.2, -0.3])

def test_normalize_shap_output_single_array_2d():
    # Typically returned by XGBRegressor or binary XGBClassifier sometimes
    # shape (n_samples, n_features)
    shap_values = np.array([[0.5, 0.6, 0.7]])
    expected_value = 0.2
    
    values, base = normalize_shap_output(shap_values, expected_value)
    
    assert base == 0.2
    assert np.allclose(values, [0.5, 0.6, 0.7])

def test_normalize_shap_output_single_array_3d():
    # Typically returned by RandomForestClassifier for multi-class or binary
    # shape (n_samples, n_features, n_classes)
    shap_values = np.array([
        [[0.1, -0.1], [0.2, -0.2], [0.3, -0.3]]
    ])
    expected_value = np.array([0.4, 0.6])
    
    values, base = normalize_shap_output(shap_values, expected_value)
    
    assert base == 0.6
    assert np.allclose(values, [-0.1, -0.2, -0.3])

def test_normalize_shap_output_list_single_class():
    # Edge case: list with 1 element
    shap_values = [
        np.array([[1.0, 2.0, 3.0]])
    ]
    expected_value = [0.8]
    
    values, base = normalize_shap_output(shap_values, expected_value)
    
    assert base == 0.8
    assert np.allclose(values, [1.0, 2.0, 3.0])

def test_normalize_shap_output_nested_base_value():
    # Edge case: deeply nested base_value
    shap_values = np.array([[0.5, 0.6, 0.7]])
    expected_value = np.array([[[[0.2]]]])
    
    values, base = normalize_shap_output(shap_values, expected_value)
    
    assert base == 0.2
