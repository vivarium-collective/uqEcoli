"""Tests for multi-parca config passthrough (PR2 §0.2)."""

import json
import tempfile
from pathlib import Path

import numpy as np

from uq.vecoli_config import _build_config, _build_variants_from_samples


class TestBuildConfigPassthrough:
    """Test _build_config preserves base config keys."""

    def test_default_config_no_base(self):
        """Without base_config_path, produces standard UQ config."""
        config = _build_config(
            sim_data_path="/tmp/simData.cPickle",
            output_dir="/tmp/output",
            variants_section={"sim_data_setattr": {}},
        )
        assert config["sim_data_path"] == "/tmp/simData.cPickle"
        assert config["emitter"] == "parquet"
        assert "parca_variants" not in config

    def test_base_config_preserved(self, tmp_path):
        """Base config keys (parca_variants, analysis_options) are preserved."""
        base = {
            "parca_variants": [
                {"rnaseq_basal_dataset_id": "glucose_minimal"},
                {"rnaseq_basal_dataset_id": "glucose_rich"},
            ],
            "analysis_options": {
                "multivariant": {"sensitivity_overview": {}}
            },
            "some_custom_key": "preserved",
        }
        base_path = tmp_path / "base.json"
        base_path.write_text(json.dumps(base))

        config = _build_config(
            sim_data_path="/tmp/simData.cPickle",
            output_dir="/tmp/output",
            variants_section={"sim_data_setattr": {}},
            base_config_path=str(base_path),
        )

        assert config["parca_variants"] == base["parca_variants"]
        assert config["analysis_options"] == base["analysis_options"]
        assert config["some_custom_key"] == "preserved"
        # UQ fields still override
        assert config["sim_data_path"] == "/tmp/simData.cPickle"
        assert config["emitter"] == "parquet"

    def test_conditions_flag_populates_parca_variants(self):
        """--conditions flag produces parca_variants in config."""
        config = _build_config(
            sim_data_path="/tmp/simData.cPickle",
            output_dir="/tmp/output",
            variants_section={"sim_data_setattr": {}},
            conditions=["glucose_minimal", "glucose_rich", "precise_wt_glc"],
        )

        assert "parca_variants" in config
        assert len(config["parca_variants"]) == 3
        assert config["parca_variants"][0] == {"rnaseq_basal_dataset_id": "glucose_minimal"}
        assert config["parca_variants"][2] == {"rnaseq_basal_dataset_id": "precise_wt_glc"}

    def test_conditions_override_base_parca_variants(self, tmp_path):
        """Explicit --conditions overrides base config parca_variants."""
        base = {
            "parca_variants": [{"rnaseq_basal_dataset_id": "old_dataset"}],
        }
        base_path = tmp_path / "base.json"
        base_path.write_text(json.dumps(base))

        config = _build_config(
            sim_data_path="/tmp/simData.cPickle",
            output_dir="/tmp/output",
            variants_section={},
            base_config_path=str(base_path),
            conditions=["new_a", "new_b"],
        )

        # conditions flag should override base parca_variants
        assert len(config["parca_variants"]) == 2
        assert config["parca_variants"][0] == {"rnaseq_basal_dataset_id": "new_a"}

    def test_no_conditions_no_parca_variants(self):
        """No conditions and no base → no parca_variants key at all."""
        config = _build_config(
            sim_data_path="/tmp/simData.cPickle",
            output_dir="/tmp/output",
            variants_section={},
        )
        assert "parca_variants" not in config

    def test_backward_compatible_with_existing_callers(self):
        """Existing callers (no base_config_path, no conditions) still work."""
        from libuq.pipeline.models import SimDataParameter

        X = np.array([[0.5, 0.1], [0.6, 0.2]])
        specs = [
            SimDataParameter(
                name="param_a",
                attr_path="process.a.val",
                bounds=(0.0, 1.0),
                description="test",
            ),
            SimDataParameter(
                name="param_b",
                attr_path="process.b.val",
                bounds=(0.0, 0.5),
                description="test",
            ),
        ]
        variants = _build_variants_from_samples(X, specs)
        config = _build_config(
            sim_data_path="/tmp/simData.cPickle",
            output_dir="/tmp/output",
            variants_section=variants,
            n_init_sims=2,
            generations=3,
        )

        assert config["n_init_sims"] == 2
        assert config["generations"] == 3
        assert "sim_data_setattr" in config["variants"]
        assert len(config["variants"]["sim_data_setattr"]["mutations"]["value"]) == 2
