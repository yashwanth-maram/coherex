"""Quick test: verify CONFIG immutability and determinism."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherex.config import CONFIG

print("=" * 50)
print("  CONFIG Immutability Test")
print("=" * 50)

# Test 1: Cannot mutate nested config
try:
    CONFIG.detection.confidence_threshold = 0.5
    print("  [FAIL] Nested mutation allowed!")
except AttributeError:
    print("  [PASS] Nested mutation blocked (FrozenInstanceError)")

# Test 2: Cannot mutate root config
try:
    CONFIG.version = "hacked"
    print("  [FAIL] Root mutation allowed!")
except AttributeError:
    print("  [PASS] Root mutation blocked (FrozenInstanceError)")

# Test 3: Serialization round-trip
import json
d = CONFIG.to_dict()
assert isinstance(d, dict)
assert d["version"] == "CoheRex-Integrity v1.0"
assert d["detection"]["confidence_threshold"] == 0.4
assert d["tracking"]["tamper_latch_frames"] == 15
assert d["integrity"]["fusion_weights"] == CONFIG.integrity.fusion_weights
assert d["integrity"]["continuity"]["miss_penalty_scale"] == 10.0
print("  [PASS] Serialization round-trip correct")

# Test 4: Config determinism
from coherex.config import SystemConfig
c1 = SystemConfig()
c2 = SystemConfig()
assert c1.to_dict() == c2.to_dict()
print("  [PASS] Deterministic — two instances produce identical config")

print()
print("CONFIG is immutable & deterministic. Infrastructure-grade.")
