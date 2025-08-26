#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Обёртка, чтобы можно было запускать:
#   python run.py ethanol.xyz --engine crest --debug
# без подкоманды "run".

from main import run as pipeline_run
import typer

if __name__ == "__main__":
    typer.run(pipeline_run)
