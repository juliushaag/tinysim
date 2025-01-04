


from pathlib import Path


OBJECT_PATH = Path(__file__).parent / "../../models/objects" 
OBJECTS = { path.name : path for path in OBJECT_PATH.iterdir() }

