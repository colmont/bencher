# Evaluating one point twice must give bit-identical values on every benchmark this
# image's Mujoco service seeds. Run inside the image by ghcr_publish.yml -- the whole
# reason for the build is that these three used to differ run to run, so an image that
# ships without this property is not worth pulling to the cluster.
import os
import types
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

from mujocobenchmarks.main import MujocoServiceServicer

# __new__ rather than __init__: binding a port is not wanted here, and the servicer's
# evaluate_point reads nothing but this one attribute.
svc = MujocoServiceServicer.__new__(MujocoServiceServicer)
svc._lunarlander_rollouts = 3  # K=3 keeps the check under a minute; seeding is K-independent

failures = []
rng = np.random.default_rng(0)
for name, dim in (("lunarlander", 12), ("mujoco-swimmer", 16), ("mujoco-hopper", 33)):
    x = rng.random(dim)
    request = types.SimpleNamespace(
        benchmark=types.SimpleNamespace(name=name),
        point=types.SimpleNamespace(values=[types.SimpleNamespace(value=float(v)) for v in x]),
    )
    first = svc.evaluate_point(request, None).value
    second = svc.evaluate_point(request, None).value
    ok = first == second
    print("%-16s %.10f %.10f %s" % (name, first, second, "OK" if ok else "NONDETERMINISTIC"))
    if not ok:
        failures.append(name)

if failures:
    raise SystemExit("::error::not deterministic: " + ", ".join(failures))
print("all deterministic")
