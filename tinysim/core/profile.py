

from dataclasses import dataclass
import time


@dataclass
class ProfileData:
  name: str
  calls: int = 0
  time_avg: float = 0 
  total_time: float = 0

class Profile():

  _PROFILES : dict[str, ProfileData]= dict()
  _REGISTERED = False

  @classmethod
  def register(cls, fn):
    profile = cls._PROFILES[fn.__name__] = ProfileData(fn.__name__)

    import atexit
    if not cls._REGISTERED:
      atexit.register(Profile._atexit)
      cls._REGISTERED = True

    def _fn_call(*args, **kwargs):
      profile.calls += 1

      start = time.monotonic()
      result = fn(*args, **kwargs)
      end = time.monotonic()

      profile.total_time += end - start
      profile.time_avg = profile.total_time / profile.calls

      return result
    
    return _fn_call

  @classmethod
  def _atexit(cls):
    if len(cls._PROFILES) == 0: return

    print("Profiling results:")
    print("-" * 40)

    for name, profile in cls._PROFILES.items():
      print(f"{name}: {profile.total_time:.2f} s ({profile.time_avg * 1000:.2f} ms avg) in {profile.calls} calls")
    print("-" * 40)
