"""Print the .env file Axolotl's CLI would load at startup, without loading it.

axolotl/cli/main.py calls python-dotenv's load_dotenv(), which searches upward from the
calling file's own directory (not the working directory) and stops at the first .env.
This runs that same search with axolotl/cli/main.py as the caller. Axolotl itself is
located, not imported. Run with a training env's python:

    python -I which_dotenv.py
"""

import importlib.util
import os

spec = importlib.util.find_spec("axolotl")
main_file = os.path.join(os.path.dirname(spec.origin), "cli", "main.py")
scope: dict = {}
# The frame's file is axolotl/cli/main.py, as in Axolotl's own call.
exec(compile("from dotenv import find_dotenv\nRESULT = find_dotenv()\n", main_file, "exec"), scope)
print(scope["RESULT"])
