"""Replay the forty saved models from the current 70/30 + three-fold protocol."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pamap2.verification_holdout import main

if __name__ == '__main__':
    main()
