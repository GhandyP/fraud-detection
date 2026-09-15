from importlib.metadata import version

import fraud_detection


def test_installed_package_identity_and_version() -> None:
    assert fraud_detection.__name__ == "fraud_detection"
    assert fraud_detection.__version__ == "0.1.0"
    assert version("fraud-detection") == fraud_detection.__version__
