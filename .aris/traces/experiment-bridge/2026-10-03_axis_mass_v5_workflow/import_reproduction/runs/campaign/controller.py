import json, sys, importlib.util
print(json.dumps({"sys_path0":sys.path[0],"project_sibling_found":importlib.util.find_spec("review_project_sibling") is not None}))
from review_project_sibling import VALUE
