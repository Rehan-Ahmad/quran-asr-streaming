#!/usr/bin/env python3
"""Wrapper that injects a minimal nv_one_logger stub then runs the
upstream `speech_to_text_finetune.py` example as __main__.
"""
import sys
import types
import runpy
from pathlib import Path
import importlib
try:
    # Some NeMo examples import nv_one_logger.pytorch_lightning_integration
    # but the installed package exposes the integration under
    # nv_one_logger.training_telemetry.integration.pytorch_lightning.
    # Alias the real module into the expected import path so imports succeed.
    pl = importlib.import_module('nv_one_logger.training_telemetry.integration.pytorch_lightning')
    sys.modules['nv_one_logger.pytorch_lightning_integration'] = pl
except Exception:
    import types
    fake_pl = types.ModuleType('nv_one_logger.pytorch_lightning_integration')
    sys.modules['nv_one_logger.pytorch_lightning_integration'] = fake_pl

# Prevent importing the real nv_one_logger training_telemetry integration module
# which may execute code incompatible with local dependency versions. Register
# a lightweight placeholder at the full package path so downstream imports use
# this no-op module instead of the installed one.
try:
    fake_full = types.ModuleType('nv_one_logger.training_telemetry.integration.pytorch_lightning')
    sys.modules['nv_one_logger.training_telemetry.integration.pytorch_lightning'] = fake_full
    # also ensure the parent package entries exist to satisfy pkgutil imports
    parent = types.ModuleType('nv_one_logger.training_telemetry.integration')
    tt_pkg = sys.modules.get('nv_one_logger.training_telemetry', types.ModuleType('nv_one_logger.training_telemetry'))
    sys.modules['nv_one_logger.training_telemetry.integration'] = parent
    sys.modules['nv_one_logger.training_telemetry'] = tt_pkg
except Exception:
    pass

# Provide minimal callback classes expected by NeMo imports so the import
# succeeds while avoiding executing the real vendor integration code.
try:
    if 'nv_one_logger.training_telemetry.integration.pytorch_lightning' in sys.modules:
        mod = sys.modules['nv_one_logger.training_telemetry.integration.pytorch_lightning']
        class TimeEventCallback:
            def __init__(self, *args, **kwargs):
                pass
        class OneLoggerPTLTrainer:
            pass
        setattr(mod, 'TimeEventCallback', TimeEventCallback)
        setattr(mod, 'OneLoggerPTLTrainer', OneLoggerPTLTrainer)
except Exception:
    pass

# Inject minimal nv_one_logger module to avoid hard dependency on vendor package
if 'nv_one_logger' not in sys.modules:
    mod = types.ModuleType('nv_one_logger')
    api_mod = types.ModuleType('nv_one_logger.api')
    config_mod = types.ModuleType('nv_one_logger.api.config')

    class OneLoggerConfig:
        def __init__(self, *args, **kwargs):
            pass

    # place OneLoggerConfig in the expected module path
    config_mod.OneLoggerConfig = OneLoggerConfig
    # attach submodules
    api_mod.config = config_mod
    mod.api = api_mod

    # Provide training_telemetry.api.callbacks with a no-op on_app_start
    tt_mod = types.ModuleType('nv_one_logger.training_telemetry')
    tt_api = types.ModuleType('nv_one_logger.training_telemetry.api')
    tt_callbacks = types.ModuleType('nv_one_logger.training_telemetry.api.callbacks')

    def on_app_start(*args, **kwargs):
        return None

    tt_callbacks.on_app_start = on_app_start
    tt_api.callbacks = tt_callbacks
    tt_mod.api = tt_api

    # attach to package
    mod.training_telemetry = tt_mod

    sys.modules['nv_one_logger'] = mod
    sys.modules['nv_one_logger.api'] = api_mod
    sys.modules['nv_one_logger.api.config'] = config_mod
    sys.modules['nv_one_logger.training_telemetry'] = tt_mod
    sys.modules['nv_one_logger.training_telemetry.api'] = tt_api
    sys.modules['nv_one_logger.training_telemetry.api.callbacks'] = tt_callbacks

# Run the original finetune script from this directory
SCRIPT = Path(__file__).with_name('speech_to_text_finetune.py')
if not SCRIPT.exists():
    raise SystemExit(f"Required script not found: {SCRIPT}")

runpy.run_path(str(SCRIPT), run_name='__main__')
