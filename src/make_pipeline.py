# make_pipeline.py
# Author: Amar Gill
# Date: Dec 11 2025

# pipeline_utils.py

from sklearn.pipeline import make_pipeline
from typing import Optional, Any
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Used since sklearn lacks ability to check transformers/model types
# Used Gemini so generate this helper function
def _has_required_methods(obj: Any, methods: list[str]) -> bool:
    """Check if an object has all the required methods."""
    return all(hasattr(obj, method) and callable(getattr(obj, method)) for method in methods)

def create_model_pipeline(
    model: Any,
    preprocessor: Optional[Any] = None
) -> Pipeline:
    """
    Constructs a sklearn Pipeline with an optional preprocessing step.

    Args:
        model: The final estimator in the pipeline.
               Must have fit and predict methods.
        preprocessor: An optional transformer (e.g., StandardScaler, ColumnTransformer)
                      to be applied before the model. If None, only the model is used.

    Returns:
        A sklearn Pipeline object

    Raises:
        TypeError: If the provided objects are not valid sklearn
                   estimators/transformers (handled by make_pipeline).
    """
    

    # 1. Validate Model: Must have 'fit' and 'predict' methods
    if not _has_required_methods(model, ['fit', 'predict']):
        raise TypeError(f"The 'model' object must be a scikit-learn estimator (requires 'fit' and 'predict' methods). Got {type(model).__name__}.")

    # Build the list of steps for the pipeline
    steps = []
    if preprocessor is not None:
        # 2. Validate Preprocessor: Must have 'fit' and 'transform' methods
        if not _has_required_methods(preprocessor, ['fit', 'transform']):
            raise TypeError(f"The 'preprocessor' object must be a scikit-learn transformer (requires 'fit' and 'transform' methods). Got {type(preprocessor).__name__}.")
        
        steps.append(preprocessor)

    steps.append(model)


    # Use the original sklearn make_pipeline to create the object
    # The unpacking of 'steps' handles the case where preprocessor is None
    return make_pipeline(*steps)

if __name__ == '__main__':
    # Example Usage:
    # 1. Pipeline with preprocessing
    pipeline_with_pre = create_model_pipeline(
        model=LogisticRegression(),
        preprocessor=StandardScaler()
    )
    print("Pipeline with Preprocessor:", pipeline_with_pre)

    # 2. Pipeline without preprocessing
    pipeline_no_pre = create_model_pipeline(
        model=LogisticRegression()
    )
    print("Pipeline without Preprocessor:", pipeline_no_pre)