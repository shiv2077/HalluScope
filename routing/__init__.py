"""Tiered risk-routing layer for model-generated content.

Modules: data (contract), generate, judge, scorers, gates, calibrate, evaluate, cost, cli.
Labels live only in data/<split>_labels.csv and are joined only inside evaluate.py and gates.py.
"""
