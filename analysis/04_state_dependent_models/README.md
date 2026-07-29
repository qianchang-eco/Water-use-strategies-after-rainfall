# 04 State-dependent models

The 21 d process table is generated with one shared script and an explicit
`--ecosystem` argument, rather than three copied ecosystem-specific scripts.
The resulting table can be passed to either the full process model or the
qoriginal-quartile model.

Example command shape:

```text
python build_rewet_process_table_21d.py --ecosystem savanna --rewet ... --first ... --pet-dir ... --output ...
python run_process_regression_21d.py --data ... --output-dir ...
python run_stratified_regression_21d.py --data ... --output-dir ...
```
