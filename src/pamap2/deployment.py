"""Export the CV-selected 70%-trained model and replay held-out segments."""
from .deployment_holdout import main, predict_demo

if __name__ == '__main__':
    main()
