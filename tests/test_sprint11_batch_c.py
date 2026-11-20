"""Sprint 11 Batch C — Dead module removal guards + midi_generator logging (Tasks 11.7-11.8)."""
import importlib.util
from pathlib import Path

import main as main_module
import pytest


class TestDeadModuleFlags:
    """Task 11.7: Dead modules permanently removed (no _deprecated/ package)."""

    def test_main_no_longer_imports_live_instrument_index(self):
        """main.py must not import the removed live instrument_index path."""
        source = Path(main_module.__file__).read_text(encoding="utf-8")
        assert "from multimodal_gen.instrument_index import" not in source

    def test_deprecated_package_removed(self):
        """multimodal_gen._deprecated and its dead modules are removed."""
        repo_root = Path(main_module.__file__).resolve().parent
        assert not (repo_root / "multimodal_gen" / "_deprecated").exists()
        for mod_name in (
            "multimodal_gen._deprecated",
            "multimodal_gen.spatial_audio",
            "multimodal_gen.instrument_registry",
            "multimodal_gen.instrument_index",
        ):
            assert importlib.util.find_spec(mod_name) is None


class TestMidiGeneratorLogging:
    """Task 11.8: midi_generator has logging for silent except blocks."""

    def test_midi_generator_has_logger(self):
        """midi_generator module has a logger configured."""
        import multimodal_gen.midi_generator as mod
        assert hasattr(mod, '_midi_logger')

    def test_logger_is_proper_type(self):
        """Logger is a standard logging.Logger instance."""
        import logging
        import multimodal_gen.midi_generator as mod
        assert isinstance(mod._midi_logger, logging.Logger)
