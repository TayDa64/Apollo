# MUSE Server Protocol Reference (OSC & JSON-RPC 2.0)

> **Version**: 1 (`SCHEMA_VERSION`)  
> **Last Updated**: Synchronized with `multimodal_gen/server/{config,osc_server,jsonrpc_server,gateway,worker}.py` and `juce/Source/Communication/Messages.h`

This document is the **canonical reference** for communication between the Python backend server (`multimodal_gen/server/`) and desktop/CLI clients (JUCE OSC client and JSON-RPC 2.0 HTTP clients).

---

## OSC Message Format

All OSC messages carry a JSON string payload (except chunked expansion instrument transfers). Each payload includes `schema_version` for compatibility checks.

```
/<address> <json_string>
```

### Schema Version Behavior

| Client Version | Server Version | Behavior |
|----------------|----------------|----------|
| Equal | Equal | ✅ Proceed normally |
| Older | Newer | ⚠️ Server warns, proceeds (backward compatible) |
| Newer | Older | ❌ Server rejects with error code `103` (`SCHEMA_VERSION_MISMATCH`) |

**Implementation**:
- Python: `multimodal_gen/server/osc_server.py` validates `schema_version` in `_validate_schema_version()` across `/generate`, `/analyze`, `/regenerate`, `/fx_chain`, `/controls/set`, and `/controls/clear`.
- JUCE: `juce/Source/Communication/Messages.h` defines `static constexpr int SCHEMA_VERSION = 1;` and `OSCBridge::Listener::onSchemaVersionWarning()`.

---

## OSC Client → Server Messages

### `/generate`
Start a music generation request.

**Payload:**
```json
{
  "schema_version": 1,
  "request_id": "uuid-string",
  "prompt": "Make a G-Funk beat at 92 BPM in A minor",
  "genre": "g_funk",
  "bpm": 92,
  "key": "A",
  "mode": "minor",
  "duration_bars": 8,
  "num_takes": 1,
  "output_dir": "",
  "soundfont": "",
  "instruments": [],
  "reference_url": "",
  "reference_bpm": 0,
  "reference_key": "",
  "reference_genre": "",
  "score_plan": null,
  "render_audio": true,
  "export_stems": false,
  "export_mpc": false,
  "verbose": false,
  "options": {
    "tension_arc_shape": "linear_build",
    "tension_intensity": 0.8,
    "motif_mode": "auto",
    "num_motifs": 2,
    "preset": "legendary",
    "style_preset": "g_funk_90s",
    "production_preset": "wide_modern",
    "orchestral_stage_preset": "epic",
    "seed": 123
  }
}
```

**Notes:**
- `prompt` is required unless a valid `score_plan` (`score_plan_v1`) supplies the prompt/plan.
- `bars` is accepted as a backward-compatible alias for `duration_bars` (sent by JUCE `GenerationRequest::toJson()`).
- Server-persisted `/controls/set` overrides are merged beneath per-request `options` (explicit request fields take precedence).

### `/cancel`
Cancel an in-progress generation.

**Payload:**
```json
{
  "schema_version": 1,
  "task_id": "optional-task-id",
  "request_id": "uuid-to-cancel"
}
```

**Notes:**
- May also be sent with no JSON arguments to cancel the active task.
- Server responds with `/status` (`status: "cancelled"`).

### `/regenerate`
Regenerate a specific bar range of the current project (sectional regeneration).

**Payload:**
```json
{
  "schema_version": 1,
  "request_id": "uuid-string",
  "start_bar": 4,
  "end_bar": 8,
  "tracks": ["drums", "bass"],
  "seed_strategy": "new",
  "prompt": "optional override prompt",
  "options": {
    "bpm": 92,
    "key": "A",
    "mode": "minor",
    "genre": "g_funk",
    "tension_arc_shape": "linear_build",
    "tension_intensity": 0.8,
    "motif_mode": "auto",
    "num_motifs": 2,
    "preset": "legendary",
    "style_preset": "g_funk_90s",
    "production_preset": "wide_modern",
    "orchestral_stage_preset": "epic",
    "seed": 123
  }
}
```

**Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `start_bar` | int | ✅ | 0-indexed starting bar |
| `end_bar` | int | ✅ | 0-indexed ending bar (exclusive) |
| `tracks` | string[] | ⚪ | Empty = all tracks; otherwise specific track names |
| `seed_strategy` | string | ⚪ | `"new"` for fresh seed, `"derived"` to vary existing |
| `prompt` | string | ⚪ | Optional override prompt for this section |
| `options` | object | ⚪ | Generation context (`bpm`, `key`, `mode`, `genre`, Phase 5.2 controls) |

### `/controls/set`
Persist control overrides on the server. Merged into subsequent `/generate` and `/regenerate` requests.

**Payload:**
```json
{
  "schema_version": 1,
  "request_id": "uuid-string",
  "overrides": {
    "tension_arc_shape": "linear_build",
    "tension_intensity": 0.8,
    "motif_mode": "on",
    "num_motifs": 2,
    "preset": "legendary",
    "style_preset": "g_funk_90s",
    "production_preset": "wide_modern",
    "orchestral_stage_preset": "epic",
    "duration_bars": 8,
    "seed": 123
  }
}
```

**Response**: `/status` with `status: "controls_updated"`, `request_id`, and `overrides`.

### `/controls/clear`
Clear persisted control overrides.

**Payload:**
```json
{
  "schema_version": 1,
  "request_id": "uuid-string",
  "keys": ["tension_intensity", "motif_mode"]
}
```

**Notes:**
- If `keys` is omitted or empty, all persisted overrides are cleared.
- **Response**: `/status` with `status: "controls_cleared"`, `request_id`, `cleared`, and `overrides`.

### `/analyze`
Analyze a local audio/MIDI file or reference URL for BPM, key, and musical characteristics.

**Payload:**
```json
{
  "schema_version": 1,
  "request_id": "uuid-string",
  "path": "/path/to/audio.wav",
  "url": "",
  "verbose": false
}
```

**Notes:**
- Provide either `path` (canonical; `file_path` is accepted as a legacy alias) OR `url`.
- Local files route to `file_analysis.py` (offline-first, supports WAV/MIDI with graceful degradation when `librosa` is absent). URLs route to `reference_analyzer.py` (requires optional `librosa`/`yt-dlp`).

### `/fx_chain`
Send FX chain configuration for offline render parity.

**Payload:**
```json
{
  "schema_version": 1,
  "request_id": "uuid-string",
  "fx_chain": {
    "master": [
      {
        "type": "eq",
        "enabled": true,
        "params": {
          "lowGain": 0.0,
          "midGain": 2.5,
          "highGain": -1.0
        }
      }
    ],
    "drums": [],
    "bass": [],
    "melodic": []
  }
}
```

**Response**: `/status` with `status: "fx_chain_updated"`.

### `/instruments`
Request a scan/refresh of available sample instruments.

**Payload:**
```json
{
  "schema_version": 1,
  "paths": ["/path/to/kit1"],
  "cache_dir": "/optional/cache/dir"
}
```

**Response**: `/status` (`status: "scan_started"`) followed asynchronously by `/instruments_loaded`.

### `/ping`
Health check heartbeat.

**Payload:**
```json
{
  "schema_version": 1
}
```

**Response**: `/pong`.

### `/shutdown`
Request graceful server shutdown.

**Payload:**
```json
{
  "schema_version": 1,
  "request_id": "uuid-string"
}
```

**Response**: `/status` with `status: "shutdown_acknowledged"`, `request_id`, and `timestamp`.

---

## Expansion Management (Client → Server)

### `/expansion/list`
List loaded expansions and categories.
- **Payload**: `{}` (optional `schema_version`)
- **Response**: `/expansion/list_response`

### `/expansion/instruments`
List instruments belonging to an expansion pack.
- **Payload**: `{"expansion_id": "funk_o_rama"}`
- **Response**: `/expansion/instruments_response` (single JSON string if `<= 8000` chars) OR chunked `/expansion/instruments_chunk` messages (`expansion_id, idx, total, chunk`) to avoid UDP datagram truncation (`WinError 10040`).

### `/expansion/resolve`
Resolve an instrument name with intelligent genre-aware matching.
- **Payload**: `{"instrument": "krar", "genre": "eskista"}`
- **Response**: `/expansion/resolve_response`

### `/expansion/import`
Import an expansion pack from disk.
- **Payload**: `{"path": "/path/to/expansion"}`
- **Response**: `/status` (`status: "expansion_imported"`, `path`)

### `/expansion/scan`
Scan a directory for expansion packs.
- **Payload**: `{"directory": "/path/to/expansions"}`
- **Response**: `/status` (`status: "expansion_scan_complete"`, `directory`, `count`)

### `/expansion/enable`
Enable or disable an expansion pack.
- **Payload**: `{"expansion_id": "funk_o_rama", "enabled": true}`
- **Response**: `/status` (`status: "expansion_enable_changed"`, `expansion_id`, `enabled`)

---

## Take Management (Client → Server)

### `/take/select`
Select a specific take for a track during audition.
```json
{
  "request_id": "uuid-string",
  "track": "drums",
  "take_id": "take_001"
}
```
**Response**: `/take/selected`.

### `/take/comp`
Composite takes across bar regions.
```json
{
  "request_id": "uuid-string",
  "track": "drums",
  "regions": [
    {"start_bar": 0, "end_bar": 4, "take_id": "take_001"},
    {"start_bar": 4, "end_bar": 8, "take_id": "take_002"}
  ]
}
```
**Response**: `/status` (`status: "comp_regions_set"`, `request_id`, `track`, `region_count`).

### `/take/render`
Render a specific take or whole-project selected/comp arrangement to WAV audio.
```json
{
  "request_id": "uuid-string",
  "track": "drums",
  "take_id": "take_001",
  "use_comp": false,
  "output_path": "/output/drums_take.wav"
}
```
**Notes:**
- Trackless `{"track": "", "take_id": "", "use_comp": true, "output_path": "/output"}` renders the latest whole-project arrangement using selected/comp tracks.
- **Response**: `/take/rendered` on success, or `/error` on failure.

---

## OSC Server → Client Messages

### `/status`
General status and command acknowledgment updates.

**Payload:**
```json
{
  "schema_version": 1,
  "status": "generating",
  "task_id": "abc12345",
  "request_id": "uuid-string",
  "prompt": "Make a G-Funk beat..."
}
```

**Observed `status` values:**
- `generating`, `regeneration_started`, `cancelled`
- `controls_updated`, `controls_cleared`, `fx_chain_updated`
- `scan_started`, `comp_regions_set`
- `expansion_imported`, `expansion_scan_complete`, `expansion_enable_changed`
- `shutdown_acknowledged`

### `/progress`
Progress updates during generation.

**Payload:**
```json
{
  "schema_version": 1,
  "request_id": "uuid-string",
  "step": "generating_midi",
  "percent": 0.35,
  "message": "Generating MIDI..."
}
```

### `/complete`
Generation completed successfully (`GenerationResult.to_dict()`).

**Payload:**
```json
{
  "schema_version": 1,
  "task_id": "abc12345",
  "request_id": "uuid-string",
  "success": true,
  "midi_path": "/path/to/output.mid",
  "audio_path": "/path/to/output.wav",
  "stems_path": "",
  "mpc_path": "",
  "render_report_path": "/path/to/output_render_report.json",
  "renderer_path": "fluidsynth",
  "pipeline_stages": {
    "fluidsynth_file_mastering.status": "applied"
  },
  "track_programs": [
    {
      "track_name": "Bass",
      "source_quality": "gm_soundfont_baseline",
      "gm_program_names": ["Electric Bass Pick"],
      "effective_programs": [34],
      "patch_family": "bass"
    }
  ],
  "metadata": {
    "bpm": 92,
    "key": "A",
    "genre": "g_funk",
    "sections": [{"name": "Intro", "bars": 4}]
  },
  "duration": 21.74,
  "samples_generated": 0,
  "instruments_used": [],
  "instrument_patches": [],
  "takes": [],
  "error_code": 0,
  "error_message": ""
}
```

### `/error`
Structured error notification.

**Payload:**
```json
{
  "schema_version": 1,
  "request_id": "uuid-string",
  "code": 200,
  "message": "Generation failed: Invalid prompt",
  "recoverable": true
}
```

### `/analyze_result`
Analysis completed with nested `analysis` object and routing metadata.

**Payload:**
```json
{
  "schema_version": 1,
  "request_id": "uuid-string",
  "success": true,
  "source_kind": "file",
  "analyzer": "file_analysis",
  "analysis": {
    "bpm": 98.5,
    "bpm_confidence": 0.85,
    "key": "G",
    "mode": "minor",
    "key_confidence": 0.72,
    "estimated_genre": "hip-hop",
    "genre_confidence": 0.68,
    "style_tags": ["punchy"]
  },
  "prompt_hints": "98 BPM hip-hop track in G minor",
  "generation_params": {
    "bpm": 98.5,
    "key": "G",
    "mode": "minor"
  },
  "warnings": []
}
```

### `/pong`
Response to `/ping`.
```json
{
  "schema_version": 1,
  "status": "ok",
  "busy": false,
  "timestamp": 1735689600.0
}
```

### `/instruments_loaded`
Response with scanned sample instruments.
```json
{
  "schema_version": 1,
  "scan_id": "abc12345",
  "success": true,
  "count": 150,
  "categories": ["drums", "bass", "keys", "synths", "strings", "brass", "fx"],
  "sources": {
    "/path/to/instruments": 120
  },
  "instruments": {}
}
```

### Expansion Responses
- `/expansion/list_response`: `{"schema_version": 1, "expansions": [...], "categories": [...]}`
- `/expansion/instruments_response`: JSON array of instrument objects (when `<= 8000` chars)
- `/expansion/instruments_chunk`: 4 OSC arguments `(expansion_id: str, idx: int, total: int, chunk: str)`
- `/expansion/resolve_response`: JSON serialization of `ResolvedInstrument.to_dict()`

### Take Responses
- `/takes/available`: `{"request_id": "...", "tracks": {...}}`
- `/take/selected`: `{"schema_version": 1, "request_id": "...", "track": "...", "take_id": "...", "success": true}`
- `/take/rendered`: `{"schema_version": 1, "request_id": "...", "track": "...", "take_id": "...", "output_path": "...", "success": true}`

### Intelligence Push Notifications (`/intel/*`)
Server-initiated push notifications emitted by `MusicGenOSCServer.notify_*`:
- `/intel/genre_dna`: `{"genre": "...", "vector": {...}}`
- `/intel/preference`: `{"dimension": "...", "old_value": 0.5, "new_value": 0.65}`
- `/intel/search`: `{"query": "...", "count": 5, "top": [{"name": "...", "score": 0.91}]}`
- `/intel/compressed`: `{"level": "2", "token_budget": 1024}`
- `/intel/fusion`: `{"name": "...", "sources": [...], "distance": "moderate"}`
- `/intel/exploration`: exploration suggestion JSON object

---

## JSON-RPC 2.0 HTTP Protocol & Dual-Protocol Gateway

In addition to OSC, MUSE exposes an HTTP JSON-RPC 2.0 server (`multimodal_gen/server/jsonrpc_server.py`, default `http://127.0.0.1:8765`, backed by `ThreadingHTTPServer`) and a dual-protocol gateway (`multimodal_gen/server/gateway.py`) sharing one `GenerationWorker` with strict task-origin callback isolation so JSON-RPC jobs do not leak unsolicited OSC notifications to JUCE.

### CLI Launch Modes (`python -m multimodal_gen.server`)
- OSC only (default): `python -m multimodal_gen.server --port 9000 --send-port 9001`
- JSON-RPC only: `python -m multimodal_gen.server --jsonrpc --port 8765`
- Dual-protocol Gateway: `python -m multimodal_gen.server --gateway --port 9000 --send-port 9001 --jsonrpc-port 8765`

### Supported JSON-RPC 2.0 Methods
| Method | Description |
|--------|-------------|
| `generate` | Submit async generation job; returns `{task_id, status}` |
| `generate_sync` | Run generation synchronously and return `GenerationResult` dict |
| `produce_sync` | Run full production workflow (optional `score_plan` + critics) synchronously |
| `get_status` | Poll task progress or cached result by `task_id` |
| `cancel` | Cancel active or specific `task_id` |
| `analyze_reference` | Analyze local audio/MIDI path or reference URL |
| `run_critics` | Run symbolic critics on inline feature arrays (`voicings`, `bass_notes`, `kick_ticks`, `tension_curve`, `note_density_per_beat`) OR on an existing `midi_path` |
| `run_critics_midi` | Extract features from a `.mid` file and run symbolic critics |
| `genre_dna_lookup` | Lookup genre DNA vector |
| `genre_blend` | Blend two or more genre DNA vectors |
| `session_state` | Return latest `SessionGraph` state |
| `midi_bridge_status` | Inspect MIDI bridge connectivity |
| `midi_bridge_play` | Stream MIDI file over MIDI bridge |
| `search_assets` / `build_index` | Semantic asset search and embedding index build |
| `get_preferences` / `record_preference` | Read or record user preference signals |
| `analyze_output` | Run post-render audio/MIDI output analysis |
| `ping` / `shutdown` | Health check and graceful server shutdown |

---

## Error Codes Reference (`multimodal_gen/server/config.py`)

| Code | Name | Description |
|------|------|-------------|
| `100` | `UNKNOWN` | Unclassified server error |
| `101` | `INVALID_MESSAGE` | Malformed JSON or invalid field value |
| `102` | `MISSING_PARAMETER` | Required field omitted |
| `103` | `SCHEMA_VERSION_MISMATCH` | Client `schema_version` is newer than server `SCHEMA_VERSION` |
| `200` | `GENERATION_FAILED` | General generation failure |
| `201` | `GENERATION_TIMEOUT` | Generation exceeded timeout |
| `202` | `GENERATION_CANCELLED` | Generation cancelled by client |
| `203` | `INVALID_PROMPT` | Prompt rejected |
| `300` | `AUDIO_RENDER_FAILED` | Offline audio rendering failed |
| `301` | `SOUNDFONT_NOT_FOUND` | Requested SoundFont does not exist |
| `302` | `FLUIDSYNTH_NOT_AVAILABLE` | FluidSynth executable unavailable |
| `400` | `INSTRUMENTS_NOT_FOUND` | No instruments found |
| `401` | `INSTRUMENT_ANALYSIS_FAILED` | Sample/instrument analysis failed |
| `500` | `FILE_NOT_FOUND` | Referenced file not found on disk |
| `501` | `FILE_WRITE_FAILED` | Failed to write output artifact |
| `502` | `OUTPUT_DIR_NOT_WRITABLE` | Output directory cannot be created or written |
| `600` | `OPTIONAL_DEPENDENCY_MISSING` | Optional library (`librosa`, `yt-dlp`, `mido`, etc.) not installed |
| `700` | `ANALYZE_FAILED` | Reference/file analysis failed |
| `900` | `SERVER_BUSY` | Server cannot accept new generation request |
| `901` | `WORKER_CRASHED` | Worker thread crashed (non-recoverable) |
| `902` | `SHUTDOWN_IN_PROGRESS` | Server is shutting down (non-recoverable) |

---

## Source Files

- **Python Server Config**: `multimodal_gen/server/config.py`
- **Python OSC Server**: `multimodal_gen/server/osc_server.py`
- **Python JSON-RPC Server**: `multimodal_gen/server/jsonrpc_server.py`
- **Python Dual-Protocol Gateway**: `multimodal_gen/server/gateway.py`
- **Python Worker**: `multimodal_gen/server/worker.py`
- **JUCE Messages**: `juce/Source/Communication/Messages.h`
- **JUCE OSC Bridge**: `juce/Source/Communication/OSCBridge.cpp`
