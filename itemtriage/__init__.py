"""itemtriage: rank dataset items (question plus answer key) for expert review under a fixed budget.

Reuses routing.stats, routing.cost and routing.policy. Labels live only in data/itemtriage/labels.parquet
and are read only through itemtriage.labels, which allows an explicit list of callers.
"""
