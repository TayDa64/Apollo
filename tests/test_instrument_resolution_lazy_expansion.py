from types import SimpleNamespace

from multimodal_gen.instrument_resolution import InstrumentResolutionService


class FakeExpansionManager:
    def __init__(self, names):
        instruments = {}
        for index, name in enumerate(names):
            instruments[str(index)] = SimpleNamespace(
                id=f"fake_{index}",
                name=name,
                path=f"/fake/{index}.wav",
            )

        self.expansions = {
            "fake": SimpleNamespace(enabled=True, instruments=instruments)
        }
        self._by_name = {
            instrument.name.lower(): instrument
            for instrument in instruments.values()
        }

    def resolve_instrument(self, requested, genre="", prefer_expansion=None):
        instrument = self._by_name.get(requested.lower())
        if not instrument:
            return SimpleNamespace(path="")

        return SimpleNamespace(
            path=instrument.path,
            name=instrument.name,
            sample_paths=[instrument.path],
            match_type=SimpleNamespace(value="exact"),
            confidence=1.0,
            source="fake",
            note="fake exact match",
        )


class MisleadingSemanticExpansionManager(FakeExpansionManager):
    def __init__(self):
        super().__init__(["Inst-Synth-Amphi Bass C0"])
        self.calls = []

    def resolve_instrument(self, requested, genre="", prefer_expansion=None):
        self.calls.append((requested, genre, prefer_expansion))
        instrument = next(iter(self._by_name.values()))
        return SimpleNamespace(
            path=instrument.path,
            name=instrument.name,
            sample_paths=[instrument.path],
            match_type=SimpleNamespace(value="semantic"),
            confidence=0.72,
            source="fake",
            note="misleading fake semantic bass/synth match",
        )


def test_constructor_indexes_expansions_without_eager_allocation_or_spam(capsys):
    manager = FakeExpansionManager([f"Inst {index}" for index in range(25)])

    service = InstrumentResolutionService(
        manager,
        auto_register_expansions=True,
    )

    captured = capsys.readouterr()
    assert "No more MIDI programs available" not in captured.out
    assert service.get_registry_stats()["expansion_programs_allocated"] == 0


def test_resolving_later_exact_expansion_name_allocates_unique_program_not_prefix():
    service = InstrumentResolutionService(
        FakeExpansionManager(["Inst 2", "Inst 20"]),
        auto_register_expansions=True,
    )

    prefix = service.resolve_instrument("Inst 2")
    later = service.resolve_instrument("Inst 20")

    assert prefix.program != later.program
    assert service.EXPANSION_PROGRAM_START <= later.program <= service.EXPANSION_PROGRAM_END
    assert service.get_program_for_instrument("Inst 20") == later.program
    assert service.get_instrument_for_program(later.program) == "inst 20"


def test_exhausted_lazy_expansion_slots_use_semantic_gm_fallback(capsys):
    names = [f"Inst {index}" for index in range(18)] + ["Wide Strings Texture"]
    service = InstrumentResolutionService(
        FakeExpansionManager(names),
        auto_register_expansions=True,
    )

    for index in range(18):
        service.resolve_instrument(f"Inst {index}")

    assert service.get_registry_stats()["expansion_programs_allocated"] == 18

    result = service.resolve_instrument("Wide Strings Texture")

    captured = capsys.readouterr()
    assert "No more MIDI programs available" not in captured.out
    assert result.name == "Wide Strings Texture"
    assert result.program == 48


def test_generic_brass_section_stays_builtin_before_expansion_semantic():
    manager = MisleadingSemanticExpansionManager()
    service = InstrumentResolutionService(
        manager,
        auto_register_expansions=True,
    )

    result = service.resolve_instrument("Brass Section", genre="cinematic")

    assert manager.calls == []
    assert result.name == "brass_section"
    assert result.program == 61
    assert result.source == "builtin"
    assert result.match_type == "default"
    assert result.program < service.EXPANSION_PROGRAM_START
    assert service.get_registry_stats()["expansion_programs_allocated"] == 0


def test_generic_strings_stay_builtin_before_expansion_semantic():
    manager = MisleadingSemanticExpansionManager()
    service = InstrumentResolutionService(
        manager,
        auto_register_expansions=True,
    )

    string_ensemble = service.resolve_instrument("string ensemble", genre="cinematic")
    strings = service.resolve_instrument("strings", genre="cinematic")

    assert manager.calls == []
    assert string_ensemble.name == "string_ensemble"
    assert string_ensemble.program == 48
    assert string_ensemble.source == "builtin"
    assert string_ensemble.match_type == "default"
    assert strings.name == "strings"
    assert strings.program == 48
    assert strings.source == "builtin"
    assert strings.match_type == "default"
    assert service.get_registry_stats()["expansion_programs_allocated"] == 0
