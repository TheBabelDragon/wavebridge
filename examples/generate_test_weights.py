from pathlib import Path
import numpy as np
Path("examples").mkdir(exist_ok=True)
rng = np.random.default_rng(7)
weights = rng.normal(0.0, 0.25, 4096).astype(np.float32)
np.save("examples/test_weights.npy", weights)
print("generated examples/test_weights.npy")
print("weights:", weights.size)
print("min:", float(weights.min()))
print("max:", float(weights.max()))
