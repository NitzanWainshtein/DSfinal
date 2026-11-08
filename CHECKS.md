# Record of executed checks

Command: `python -m pytest checks -v` (Python 3.13.18, 2026-11-08). Result: **32 passed in 13.21s**.

| Check | What it establishes | Outcome |
|---|---|---|
| `test_app.py::test_screens_render_without_errors[Data overview]` | Each of the four screens renders without exceptions | PASSED |
| `test_app.py::test_screens_render_without_errors[Forecast one hour]` | Each of the four screens renders without exceptions | PASSED |
| `test_app.py::test_screens_render_without_errors[Compare models]` | Each of the four screens renders without exceptions | PASSED |
| `test_app.py::test_screens_render_without_errors[Inspect model behaviour]` | Each of the four screens renders without exceptions | PASSED |
| `test_app.py::test_forecast_reveal_and_agreement` | Forecast hidden outcome until Reveal; app forecast equals saved prediction; error correct | PASSED |
| `test_app.py::test_switching_model_clears_forecast_and_baseline_has_no_intervals` | Changing model clears forecast/reveal; baseline intervals 'unavailable' | PASSED |
| `test_features.py::test_target_hour_outcome_and_weather_do_not_change_inputs` | Changing the target hour's count/weather leaves its inputs unchanged | PASSED |
| `test_features.py::test_later_hours_do_not_change_inputs` | Changing later hours leaves the inputs unchanged | PASSED |
| `test_features.py::test_earlier_eligible_count_changes_its_input` | An eligible earlier count does affect the matching input | PASSED |
| `test_features.py::test_missing_previous_timestamp_is_missing_not_shifted` | Absent t-1 is flagged missing; the row above in the file is not used | PASSED |
| `test_features.py::test_real_missing_hour_2011_01_02_05` | Real gap: 06:00 forecast on 2011-01-02 has missing previous-hour count | PASSED |
| `test_features.py::test_observed_zero_is_kept` | A recorded zero stays zero with flag false | PASSED |
| `test_features.py::test_24h_mean_requires_complete_window` | Window 1..24 gives 12.5; one absent hour makes it missing | PASSED |
| `test_features.py::test_fast_builder_matches_explicit` | Vectorised builder equals the explicit builder | PASSED |
| `test_features.py::test_brief_worked_example_inputs` | 2012-08-01 08:00 inputs equal the brief's table | PASSED |
| `test_features.py::test_missing_inputs_use_training_median_and_flag_before_replacement` | Missing inputs receive stored medians; flags set before replacement | PASSED |
| `test_features.py::test_unseen_category_maps_to_zero_block` | Unseen category -> all-zero one-hot block | PASSED |
| `test_frozen.py::test_changing_test_values_leaves_preprocessing_unchanged` | Random test values: medians, encoding, scaling unchanged | PASSED |
| `test_frozen.py::test_changing_test_values_leaves_coefficients_and_default_unchanged` | Random test values: coefficients, validation table, selections, default unchanged | PASSED |
| `test_frozen.py::test_saved_models_reproduce_saved_predictions` | Reloaded models reproduce predictions.csv within tolerance | PASSED |
| `test_frozen.py::test_saved_files_consistent` | All models scored on the same observed test hours; intervals only for Poisson/Bayesian; metrics match predictions | PASSED |
| `test_frozen.py::test_audit_finds_brief_missing_hour` | Audit finds 2011-01-02 05:00 missing; 17,379 rows kept | PASSED |
| `test_frozen.py::test_duplicate_and_invalid_rows_are_excluded` | Duplicate (keep smallest instant) and negative-count rows excluded with reasons | PASSED |
| `test_metrics.py::test_mae_rmse_three_row_example` | MAE = 2, RMSE = 2.160 on the brief's example | PASSED |
| `test_metrics.py::test_coverage_inclusive_endpoints_two_of_three` | Coverage 2/3 with an outcome on an endpoint; width separately | PASSED |
| `test_metrics.py::test_poisson_quantiles_against_cdf` | Poisson quantiles satisfy the CDF definition | PASSED |
| `test_metrics.py::test_poisson_log_score_matches_formula` | Poisson log score equals mu - y log mu + log y! | PASSED |
| `test_metrics.py::test_bayesian_small_example` | Brief example: m=(1.6,1.2), S, mean 4, coef var 2, total 3, 80% [1.78, 6.22]; S symmetric | PASSED |
| `test_metrics.py::test_bayesian_matches_independent_inverse_and_nonnegative_variance` | Linear solves match an explicit inverse; predictive variances non-negative | PASSED |
| `test_metrics.py::test_gaussian_interval_illustration` | Mean 100, sd 10: 80% [87.18, 112.82], 95% [80.40, 119.60] | PASSED |
| `test_metrics.py::test_seasonal_baseline_toy_and_fallback` | Toy groups give 120 / 30; unseen pair uses overall mean; no intervals | PASSED |
| `test_metrics.py::test_clipping_keeps_raw_mean` | Negative raw mean kept; point forecast clipped at 0 | PASSED |

Manual check: `streamlit run app.py --server.headless true` started and `/_stcore/health` returned `ok`.
