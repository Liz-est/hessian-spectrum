#!/usr/bin/env python
"""SGD run: same tiny transformer / dataset as the Adam run, but SGD(momentum=0.9)
with a much larger LR (SGD needs 0.1~1.0 on this task; 6e-4 would look like
"SGD can't learn"). Separate out/log/comment so it doesn't clobber the Adam run.

Self-contained submission entrypoint: chdir()s into its own folder first, so
train_gpt2.py's relative paths resolve regardless of the launch directory.

Submit this file directly, e.g.:
    python run_tiny_sgd.py
"""
import os
import sys
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

sys.argv = [
    "train_gpt2.py",
    "config/train_gpt2_tiny.py",
    "--dataset=synth_uniform_balanced",
    "--use_sgd=True",
    "--learning_rate=0.3",
    "--comment=tiny_sgd",
    "--save_dir=log_gpt2/tiny_sgd",
    "--out_dir=out-gpt2/tiny_sgd",
]

runpy.run_path(os.path.join(HERE, "train_gpt2.py"), run_name="__main__")
