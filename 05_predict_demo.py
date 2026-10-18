from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from pamap2.common import seed_all
from pamap2.deployment import predict_demo
if __name__ == '__main__':
    seed_all(42)
    predict_demo()
