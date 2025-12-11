# make_pipeline_tests.py generated from Gemini AI

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
from sklearn.dummy import DummyClassifier
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.make_pipeline import create_model_pipeline 


# 1. Fixture to provide standard test data
@pytest.fixture
def sample_data():
    """Provides sample features (X) and target (y) arrays."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    return X, y

# 2. Functional Test: Testing the end-to-end usage with real objects
def test_pipeline_functionality_fit_predict(sample_data):
    """Test that the resulting pipeline can actually be fitted and used."""
    X, y = sample_data
    
    # Use real scikit-learn estimators for functional testing
    model = LogisticRegression(random_state=42)
    preprocessor = StandardScaler()

    pipeline = create_model_pipeline(model=model, preprocessor=preprocessor)

    # Ensure it's not fitted initially
    with pytest.raises(NotFittedError):
        pipeline.predict(X)

    # Fit the pipeline
    pipeline.fit(X, y)

    # Predict and check the output shape
    predictions = pipeline.predict(X)
    assert predictions.shape == y.shape
    assert hasattr(pipeline, 'classes_') # Check model attribute after fitting

# 3. Mocked Tests: Using the 'mocker' fixture for isolated testing

def test_pipeline_creation_with_preprocessor(mocker):
    """Test that the pipeline is created correctly with a preprocessor."""
    # The 'mocker' fixture replaces MagicMock usage
    mock_model = mocker.MagicMock(spec=LogisticRegression)
    mock_preprocessor = mocker.MagicMock(spec=StandardScaler)

    pipeline = create_model_pipeline(
        model=mock_model,
        preprocessor=mock_preprocessor
    )

    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2

    # Check that the mocked objects are in the pipeline steps
    assert pipeline.steps[0][1] is mock_preprocessor
    assert pipeline.steps[1][1] is mock_model

def test_pipeline_creation_without_preprocessor(mocker):
    """Test that the pipeline is created correctly without a preprocessor."""
    mock_model = mocker.MagicMock(spec=LogisticRegression)
    
    pipeline = create_model_pipeline(
        model=mock_model
    )

    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 1
    assert pipeline.steps[0][1] is mock_model

# 4. Error/Edge Case Tests

def test_invalid_model_type():
    """Test that a TypeError is raised when the model is not a valid estimator (lacks 'fit' or 'predict')."""
    # Using a simple string which lacks the required methods
    with pytest.raises(TypeError, match="must be a scikit-learn estimator"):
        create_model_pipeline(model="not_an_estimator", preprocessor=StandardScaler())

def test_invalid_preprocessor_type():
    """Test that a TypeError is raised when the preprocessor is not a valid transformer (lacks 'fit' or 'transform')."""
    # Using an integer which lacks the required methods
    with pytest.raises(TypeError, match="must be a scikit-learn transformer"):
        create_model_pipeline(model=LogisticRegression(), preprocessor=12345)

def test_with_dummy_classifier():
    """Test that it works with another valid estimator."""
    pipeline = create_model_pipeline(model=DummyClassifier(strategy="most_frequent"))
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 1
    assert 'dummyclassifier' in pipeline.named_steps