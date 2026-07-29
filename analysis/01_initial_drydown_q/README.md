# 01 Initial drydown and q

This stage identifies soil-moisture drydown events and estimates the nonlinear
water-use strategy parameter `q` for each event.

Run `detect_initial_drydowns_all_files.py` before
`fit_initial_q_all_events.py`. The lower-level functions live in
`detect_drydown_events.py` and `fit_q_event.py`.
