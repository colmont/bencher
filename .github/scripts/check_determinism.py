# What this image promises about the Mujoco service, checked inside the image by
# ghcr_publish.yml. Two properties, and the build is not worth shipping without either:
#
#   1. Seeding works. The same point at the same random_seed returns a bit-identical
#      value, and a different seed returns a different one. These benchmarks used to
#      differ run to run with no way to pin them down.
#   2. The per-request config is actually read. A server built against a scaffold that
#      predates the config field drops it as an unknown protobuf field and quietly
#      answers with its own rollout count, which is worse than an error because the
#      numbers look entirely plausible.
import os
import types
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

from mujocobenchmarks.main import MujocoServiceServicer

# __new__ rather than __init__: binding a port is not wanted here, and evaluate_point
# reads no instance state.
svc = MujocoServiceServicer.__new__(MujocoServiceServicer)


class Aborted(Exception):
    """Stands in for the error a real gRPC context.abort() raises."""


class Context:
    """Duck-typed grpc.ServicerContext: abort() raises rather than returning."""

    def abort(self, code, details):
        raise Aborted("%s: %s" % (code, details))


class Request:
    """Duck-typed BenchmarkRequest: enough of the protobuf surface for evaluate_point."""

    def __init__(self, name, x, random_seed=None, config=None):
        self.benchmark = types.SimpleNamespace(name=name)
        self.point = types.SimpleNamespace(
            values=[types.SimpleNamespace(value=float(v)) for v in x]
        )
        self._seeded = random_seed is not None
        self.random_seed = 0 if random_seed is None else random_seed
        self.config = dict(config or {})

    def HasField(self, field):  # noqa: N802 - protobuf's spelling
        if field == "random_seed":
            return self._seeded
        raise AttributeError(field)


def value(name, x, random_seed=None, config=None):
    return svc.evaluate_point(Request(name, x, random_seed, config), Context()).value


failures = []


def check(label, ok, detail=""):
    print("%-58s %s   %s" % (label, "OK" if ok else "FAILED", detail))
    if not ok:
        failures.append(label)


ONE = {"num_rollouts": "1"}
rng = np.random.default_rng(0)
for name, dim in (("lunarlander", 12), ("mujoco-swimmer", 16), ("mujoco-hopper", 33)):
    x = rng.random(dim)

    same_a = value(name, x, random_seed=17, config=ONE)
    same_b = value(name, x, random_seed=17, config=ONE)
    check("%s: same seed is bit-identical" % name, same_a == same_b,
          "%.10f %.10f" % (same_a, same_b))

    other = value(name, x, random_seed=18, config=ONE)
    check("%s: a different seed gives a different value" % name, same_a != other,
          "%.10f %.10f" % (same_a, other))

    # If config were being dropped, both of these would be the server's own default and
    # would therefore come out equal.
    two = value(name, x, random_seed=17, config={"num_rollouts": "2"})
    check("%s: num_rollouts is honoured" % name, same_a != two,
          "K=1 %.10f  K=2 %.10f" % (same_a, two))

    try:
        value(name, x, random_seed=17, config={"no_such_option": "1"})
        check("%s: an unsupported option is rejected" % name, False, "accepted it")
    except Aborted as error:
        check("%s: an unsupported option is rejected" % name, True, str(error))

    # Unseeded requests are meant to stay noisy; reported, not asserted, because
    # "these two floats differ" is not something to fail a build on.
    print("%-58s %.10f %.10f" % ("%s: unseeded (expected to vary)" % name,
                                 value(name, x), value(name, x)))

if failures:
    raise SystemExit("::error::" + "; ".join(failures))
print("all checks passed")
