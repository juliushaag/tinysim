from dataclasses import dataclass

@dataclass
class Actuator:
  name: str
  type: str
  joint: str
