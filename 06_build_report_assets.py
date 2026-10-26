"""Generate current paper tables from audited 70/30 and training-CV records."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from pamap2.report_holdout import main

if __name__ == '__main__':
    main()
