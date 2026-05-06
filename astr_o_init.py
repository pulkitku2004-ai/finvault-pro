"""
FinVault — ASTR-O pipeline singleton.

Adds the ASTR-O repo root to sys.path so its flat-module sprint files
(metadata_bridge, contradiction_detector, …) are importable, then
constructs one ASTROPipeline instance for the lifetime of the process.
"""

import os
import sys

# Make the ASTR-O repo importable (the package lives at ~/astr-o/astr_o/)
_ASTRO_REPO = os.path.expanduser("~/astr-o")
if _ASTRO_REPO not in sys.path:
    sys.path.insert(0, _ASTRO_REPO)

from astr_o.pipeline import ASTROPipeline          # type: ignore[import]  # noqa: E402
from astr_o.trust_surface.response_formatter import format_response  # type: ignore[import]  # noqa: E402, F401
from finvault_domain_schema import FINVAULT_SCHEMA  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))

_REGISTRY_PATH  = os.path.join(_HERE, "reference_registry.json")
_HOT_STORAGE    = os.path.join(_HERE, "astr_o_data", "hot")
_COLD_STORAGE   = os.path.join(_HERE, "astr_o_data", "cold")
_MISSION_ID     = "FINVAULT_HDFC_Q3_FY2025"

os.makedirs(_HOT_STORAGE,  exist_ok=True)
os.makedirs(_COLD_STORAGE, exist_ok=True)

_pipeline: ASTROPipeline | None = None


def get_pipeline() -> ASTROPipeline:
    """Return the process-level ASTROPipeline singleton (lazy-initialised)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ASTROPipeline(
            registry_path=_REGISTRY_PATH,
            domain_schema=FINVAULT_SCHEMA,
            hot_storage_path=_HOT_STORAGE,
            cold_storage_path=_COLD_STORAGE,
            mission_id=_MISSION_ID,
            enable_dashboard=True,
        )
    return _pipeline
