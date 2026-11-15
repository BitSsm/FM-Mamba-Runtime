"""Make the installed binary helpers available beside the readable engine."""
import sys
from fm_mamba_runtime import engine_directory
binary_engine = engine_directory('pcd')
if str(binary_engine) not in sys.path:
    sys.path.append(str(binary_engine))
