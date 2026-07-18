#!/usr/bin/env python
"""Adam run: tiny transformer on synth_uniform_balanced, weight tying disabled.

Self-contained submission entrypoint. Unlike run_tiny_adam.sh, this file does
NOT depend on the caller's working directory: it chdir()s into its own folder
(language_models/test_data/) first, so the relative paths that train_gpt2.py
expects (config/, configurator.py, ../../data..., files/) resolve correctly no
matter where the platform launches it from.

Submit this file directly, e.g.:
    python run_tiny_adam.py
"""
import os
import sys
import runpy

# Make relative paths inside train_gpt2.py resolve against this script's dir.
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# Reproduce exactly what run_tiny_adam.sh passed on the command line.
# argv[0] is the "program name"; argv[1] is the config file; the rest are
# --key=value overrides consumed by configurator.py inside train_gpt2.py.
sys.argv = [
    "train_gpt2.py",
    "config/train_gpt2_tiny.py",
    "--dataset=synth_uniform_balanced",
    "--use_sgd=False",
    "--learning_rate=6e-4",
    "--comment=tiny_adam",
    "--save_dir=log_gpt2/tiny_adam",
    "--out_dir=out-gpt2/tiny_adam",
]

# Run train_gpt2.py as __main__ in this process (equivalent to `python -u train_gpt2.py ...`).
runpy.run_path(os.path.join(HERE, "train_gpt2.py"), run_name="__main__")
