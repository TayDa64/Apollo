"""
Synthesizer Interface Package

Provides an abstraction layer for audio synthesis backends.
Enables swapping synth engines (FluidSynth, Procedural, VST, etc.)
without modifying the AudioRenderer.

Usage:
    from multimodal_gen.synthesizers import (
        ISynthesizer,
        SynthesizerFactory,
        SynthNote,
        SynthTrack,
        SynthResult,
    )
    
    # Get best available synthesizer
    synth = SynthesizerFactory.get_best_available(prefer_soundfont=True)
    
    # Or create specific synthesizer
    synth = SynthesizerFactory.create('procedural', genre='trap')
"""

from .base import (
    ISynthesizer,
    SynthNote,
    SynthTrack,
    SynthResult,
)
from .factory import SynthesizerFactory
from .fluidsynth_synth import FluidSynthSynthesizer
from .neural_runtime import (
    NEURAL_CONTRACT_VERSION,
    AudioConditioningRef,
    InstrumentPatchV1,
    LocalGGUFNeuralAdapter,
    NeuralAudioRenderRequest,
    NeuralAudioRenderResponse,
    NeuralBackendConfig,
    NeuralBackendStatus,
    NeuralGenerationRequest,
    NeuralGenerationResponse,
    NeuralRenderResult,
    NeuralRuntimeContractV1,
    NeuralRuntimeProtocol,
    OptionalNeuralRuntime,
    RemoteOpenAICompatibleNeuralAdapter,
    ScorePlanV1,
    create_neural_runtime,
    export_neural_runtime_json_schema,
)
from .procedural_synth import ProceduralSynthesizer
from .plugin_synth import PluginSynthesizer

__all__ = [
    # Base interface
    'ISynthesizer',
    'SynthNote',
    'SynthTrack',
    'SynthResult',
    # Factory
    'SynthesizerFactory',
    # Implementations
    'FluidSynthSynthesizer',
    'NEURAL_CONTRACT_VERSION',
    'AudioConditioningRef',
    'InstrumentPatchV1',
    'LocalGGUFNeuralAdapter',
    'NeuralAudioRenderRequest',
    'NeuralAudioRenderResponse',
    'NeuralBackendConfig',
    'NeuralBackendStatus',
    'NeuralGenerationRequest',
    'NeuralGenerationResponse',
    'NeuralRenderResult',
    'NeuralRuntimeContractV1',
    'NeuralRuntimeProtocol',
    'OptionalNeuralRuntime',
    'RemoteOpenAICompatibleNeuralAdapter',
    'ScorePlanV1',
    'create_neural_runtime',
    'export_neural_runtime_json_schema',
    'ProceduralSynthesizer',
    'PluginSynthesizer',
]
