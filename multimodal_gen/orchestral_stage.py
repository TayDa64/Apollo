"""Data-driven stereo stage profiles for orchestral MIDI rendering.

The profile is intentionally small and serializable so later UI/preset work can
surface it without rewriting MIDI generation.  Values are applied to the
FluidSynth/offline MIDI path via GM-compatible CC10 (pan), CC7 (volume), CC11
(expression), and CC74 (brightness/presence).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Sequence


@dataclass(frozen=True)
class StageSlot:
    """Stereo/level defaults for one orchestral role."""

    role: str
    pan: float  # -1.0 left .. 0 center .. +1.0 right
    cc7_volume: int = 100
    cc11_expression: int = 112
    cc74_brightness: int = 90
    width: float = 0.5  # reserved for later post-render/user controls
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StageRide:
    """Section-level presence ride offsets for orchestral roles."""

    cc7_delta: int = 0
    cc11_delta: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OrchestralStageProfile:
    """Serializable orchestral stereo stage profile."""

    name: str
    slots: Mapping[str, StageSlot]
    section_rides: Mapping[str, StageRide]
    notes: str = ""

    def get_slot(self, role: str) -> StageSlot:
        key = normalize_stage_role(role)
        return self.slots.get(key, self.slots["default"])

    def get_ride(self, section_type: str) -> StageRide:
        key = normalize_section_type(section_type)
        return self.section_rides.get(key, self.section_rides.get("default", StageRide()))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "notes": self.notes,
            "slots": {key: slot.to_dict() for key, slot in self.slots.items()},
            "section_rides": {key: ride.to_dict() for key, ride in self.section_rides.items()},
        }


def _clamp_midi(value: int) -> int:
    return max(0, min(127, int(value)))


def normalize_stage_role(role: str) -> str:
    key = str(role or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "chords": "strings",
        "string_ensemble": "strings",
        "string_ensemble_1": "strings",
        "lead": "melody",
        "trumpet": "melody",
        "french_horn": "melody",
        "horn": "melody",
        "brass_section": "brass",
        "percussion": "timpani",
    }
    return aliases.get(key, key or "default")


def normalize_section_type(section_type: str) -> str:
    key = str(section_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "intro": "intro",
        "verse": "verse",
        "underscore": "verse",
        "development": "buildup",
        "build": "buildup",
        "build_up": "buildup",
        "pre_chorus": "buildup",
        "chorus": "chorus",
        "climax": "chorus",
        "drop": "chorus",
        "bridge": "bridge",
        "variation": "buildup",
        "outro": "outro",
        "ending": "outro",
    }
    return aliases.get(key, key or "default")


def midi_pan_from_unit(pan: float) -> int:
    """Map -1..+1 pan to MIDI CC10 0..127 with center near 64."""
    p = max(-1.0, min(1.0, float(pan)))
    return max(0, min(127, int(round((p + 1.0) * 63.5))))


def apply_stage_ride(slot: StageSlot, ride: StageRide) -> tuple[int, int]:
    """Return (CC7, CC11) after a gentle section ride."""
    return _clamp_midi(slot.cc7_volume + ride.cc7_delta), _clamp_midi(slot.cc11_expression + ride.cc11_delta)


def _slot(
    role: str,
    pan: float,
    cc7: int,
    cc11: int,
    cc74: int,
    width: float,
    notes: str,
) -> StageSlot:
    return StageSlot(role=role, pan=pan, cc7_volume=cc7, cc11_expression=cc11, cc74_brightness=cc74, width=width, notes=notes)


def _rides(scale: int = 1) -> Dict[str, StageRide]:
    # Subtle offsets: support beds ease back in intro/verse/outro and come
    # forward during buildup/chorus without pump-gain behavior.
    return {
        "intro": StageRide(cc7_delta=-3 * scale, cc11_delta=-5 * scale),
        "verse": StageRide(cc7_delta=-2 * scale, cc11_delta=-4 * scale),
        "buildup": StageRide(cc7_delta=1 * scale, cc11_delta=3 * scale),
        "bridge": StageRide(cc7_delta=0, cc11_delta=1 * scale),
        "chorus": StageRide(cc7_delta=2 * scale, cc11_delta=5 * scale),
        "outro": StageRide(cc7_delta=-3 * scale, cc11_delta=-5 * scale),
        "default": StageRide(cc7_delta=0, cc11_delta=0),
    }


CHAMBER_STAGE = OrchestralStageProfile(
    name="chamber",
    notes="Intimate/narrow stereo orchestral seating with slightly restrained bed levels.",
    section_rides=_rides(scale=1),
    slots={
        "strings": _slot("strings", -0.18, 106, 112, 90, 0.45, "Narrow left string bed."),
        "brass": _slot("brass", 0.18, 104, 110, 92, 0.35, "Small brass support, modest right."),
        "harp": _slot("harp", -0.34, 104, 110, 92, 0.25, "Focused chamber harp left."),
        "choir": _slot("choir", 0.03, 102, 108, 86, 0.45, "Small choir near center."),
        "timpani": _slot("timpani", -0.03, 108, 112, 86, 0.15, "Near-center low percussion."),
        "melody": _slot("melody", 0.10, 106, 112, 92, 0.20, "Focused lead near center."),
        "default": _slot("default", 0.0, 104, 110, 88, 0.35, "Centered fallback."),
    },
)


CONCERT_STAGE = OrchestralStageProfile(
    name="concert",
    notes=(
        "Stereo-only orchestral seating: strings left-of-center, harp left, "
        "brass/lead right-of-center, choir near center, timpani near center. "
        "Depth remains controlled by level/expression and file-level reverb."
    ),
    section_rides=_rides(scale=1),
    slots={
        "strings": _slot("strings", -0.30, 112, 118, 94, 0.75, "String ensemble bed, slightly left and wider."),
        "brass": _slot("brass", 0.32, 110, 116, 98, 0.55, "Brass support right-of-center with extra presence."),
        "harp": _slot("harp", -0.55, 108, 114, 96, 0.35, "Harp off-left but focused."),
        "choir": _slot("choir", 0.08, 106, 112, 90, 0.80, "Choir near center/back; width reserved for post-render staging."),
        "timpani": _slot("timpani", -0.05, 112, 116, 88, 0.20, "Timpani/percussion stays near center for low-end stability."),
        "melody": _slot("melody", 0.18, 110, 116, 96, 0.25, "Focused orchestral lead slightly right-of-center."),
        "default": _slot("default", 0.0, 108, 114, 92, 0.50, "Fallback centered role."),
    },
)


EPIC_STAGE = OrchestralStageProfile(
    name="epic",
    notes="Wide trailer-style orchestral seating with broader lateral contrast and stronger lead/bed separation.",
    section_rides=_rides(scale=1),
    slots={
        "strings": _slot("strings", -0.45, 115, 121, 96, 0.90, "Wide left string bed."),
        "brass": _slot("brass", 0.46, 113, 120, 101, 0.75, "Wide/right heroic brass."),
        "harp": _slot("harp", -0.68, 108, 114, 98, 0.40, "Far-left sparkle, still bounded."),
        "choir": _slot("choir", 0.00, 110, 116, 92, 0.95, "Centered wide choir mass."),
        "timpani": _slot("timpani", -0.02, 114, 119, 90, 0.25, "Near-center impact."),
        "melody": _slot("melody", 0.30, 113, 120, 99, 0.30, "Forward lead/horn right-of-center."),
        "default": _slot("default", 0.0, 110, 116, 94, 0.60, "Centered fallback."),
    },
)


DEFAULT_ORCHESTRAL_STAGE = CONCERT_STAGE


STAGE_PROFILES: Dict[str, OrchestralStageProfile] = {
    "chamber": CHAMBER_STAGE,
    "intimate": CHAMBER_STAGE,
    "small": CHAMBER_STAGE,
    "concert": CONCERT_STAGE,
    "standard": CONCERT_STAGE,
    "cinematic_default_stereo_stage": CONCERT_STAGE,
    "cinematic": CONCERT_STAGE,
    "classical": CONCERT_STAGE,
    "orchestral": CONCERT_STAGE,
    "epic": EPIC_STAGE,
    "wide": EPIC_STAGE,
    "trailer": EPIC_STAGE,
    "huge": EPIC_STAGE,
}


CHAMBER_HINTS = ("chamber", "intimate", "small ensemble", "small orchestra", "close", "delicate")
EPIC_HINTS = ("epic", "wide", "trailer", "huge", "massive", "cinematic trailer", "heroic")


def resolve_stage_profile_name(
    *,
    genre: Optional[str] = None,
    prompt: Optional[str] = None,
    instruments: Optional[Sequence[str]] = None,
) -> str:
    """Resolve a deterministic stage preset name from bounded prompt hints."""
    text = " ".join(
        [str(genre or ""), str(prompt or "")] + [str(inst or "") for inst in (instruments or [])]
    ).strip().lower().replace("-", " ")
    if any(hint in text for hint in EPIC_HINTS):
        return "epic"
    if any(hint in text for hint in CHAMBER_HINTS):
        return "chamber"
    return "concert"


def get_stage_profile(name: str = "concert") -> OrchestralStageProfile:
    return STAGE_PROFILES.get(str(name or "").strip().lower(), CONCERT_STAGE)


def get_stage_slot(role: str, profile_name: str = "concert") -> StageSlot:
    return get_stage_profile(profile_name).get_slot(role)
