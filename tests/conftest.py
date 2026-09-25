import os
import sys

# Import the in-tree package (dev builds put the native modules in pybattle/), not an installed copy.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
