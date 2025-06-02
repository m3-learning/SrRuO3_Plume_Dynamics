import sys
import os

# Add _extensions to sys.path explicitly
sys.path.insert(0, os.path.abspath("./_extensions"))

# Load your custom extension
extensions = ["suppfigure"]
