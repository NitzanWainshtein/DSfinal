# AI development log

The brief (page 24) requires at least six records: two **design**, two **implementation**, two **checking/debugging**, each with task, prompt, response excerpt, decision, code/change reference and executed verification. At least two must describe a proposal that was **rejected or corrected**.

The records below are factual entries from the first development session (Claude, claude.ai, 8 Nov 2026). The *Decision* fields marked **TODO** must be completed by the team in your own words after you have reviewed the code. Add further records from your own later sessions; keep only records you can stand behind.

---

## D1 - Design: choosing Option B and the module layout

* **Task:** choose between Project A and Project B and plan the code organisation.
* **Prompt (summary):** shared the course page and asked to start the final project.
* **Response excerpt:** compared both briefs (data, methods, screens, LLM dependency) and recommended B as "fewer moving parts, no API dependency, closer to the lecture material"; proposed `data.py` / `features.py` / `models.py` / `train_evaluate.py` / `app.py` / `checks/`.
* **Decision:** chose Option B. **TODO:** state why the team agreed (or what you changed).
* **Reference:** repository layout, first commit.
* **Verification:** layout matches brief page 22.

## D2 - Design: historical inputs by exact timestamp lookup

* **Task:** decide how lag features are built so that missing hours cannot leak or shift.
* **Prompt (summary):** implement the page 6 inputs with the page 4 audit rules.
* **Response excerpt:** used `clean[col].reindex(t - k hours)` (exact lookup) instead of `shift(k)`, plus a rolling 24-hour mean over the *complete* hourly calendar with `min_periods=24`.
* **Decision:** **TODO.**
* **Reference:** `features.py` (`build_raw_features`, `build_raw_features_fast`).
* **Verification:** `checks/test_features.py::test_missing_previous_timestamp_is_missing_not_shifted`, `test_real_missing_hour_2011_01_02_05`, `test_24h_mean_requires_complete_window`, `test_brief_worked_example_inputs` all pass (the 2012-08-01 08:00 example reproduces 487, 760, 788, 300.4167, 0.64, 0.83, 0.1343).

## I1 - Implementation: data source (correction)

* **Task:** download `hour.csv`.
* **Response excerpt:** the first attempt to download from archive.ics.uci.edu was refused by the sandbox network; the assistant fetched the file from a GitHub copy instead.
* **Correction / decision:** a mirror is not automatically trustworthy, so the file was verified before use. **TODO:** confirm you are happy with this, or re-download from UCI and compare the hash in the README.
* **Reference:** `data/hour.csv`, README "Data".
* **Verification:** 17,379 rows; values quoted in the brief (2012-08-01 07:00 = 487, 08:00 = 681; 2011-01-02 05:00 absent) checked by script and by `checks/test_frozen.py::test_audit_finds_brief_missing_hour`.

## I2 - Implementation: penalty grids (correction)

* **Task:** fit Ridge, Lasso, Poisson and Bayesian candidates and select by validation MAE.
* **Response excerpt (first version):** grids Ridge {0.1 ... 1000}, Lasso {0.01 ... 30}, Poisson {1e-5 ... 0.1}, Bayesian alpha {1e-6 ... 1}.
* **Problem found:** every method selected the *smallest* value in its grid, so the optimum might lie outside the grid.
* **Correction:** grids extended downwards (Ridge to 0.001, Lasso to 0.001, Poisson to 1e-7, Bayesian alpha to 1e-8). Validation MAE then flattens (Ridge 50.228 at 0.001 and 0.01), showing a plateau rather than a missed optimum. **TODO:** decision in your words.
* **Reference:** `train_evaluate.py` `GRIDS`; `outputs/validation_results.csv`.
* **Verification:** rerun of `train_evaluate.py`; Regularisation panel in the app.

## C1 - Checking: Poisson errors much larger than the linear models

* **Task:** explain Poisson validation MAE 84.0 vs about 50.2 for Ridge.
* **Response excerpt:** checked convergence (`n_iter_` = 236 of 5000, converged) and inspected the largest forecasts: evening peaks such as 2012-09-11 18:00 forecast 2531 against 877 recorded. With a log link, the standardised lag counts act multiplicatively, so the higher 2012 demand is extrapolated exponentially.
* **Decision:** kept the model, because the brief requires the same transformed input vector for all regression methods; documented it as a failure mode with a proposed change (log-transformed lag counts for the Poisson model, tested on validation). **TODO:** confirm.
* **Reference:** `models.py::PoissonModel`; report "Limitations".
* **Verification:** saved predictions inspected; Poisson coverage 23.5% at nominal 80% recorded in `metrics.csv`.

## C2 - Checking: proving learned values do not depend on test outcomes

* **Task:** implement the page 23 check that changing test values cannot alter preprocessing, coefficients or the default.
* **Response excerpt:** refactored steps 2-4 of `train_evaluate.py` into `fit_select(clean)`, then wrote a check that reruns it on data whose test-period `cnt` and `temp` were replaced by random values and asserts identical preprocessing, coefficients, validation table and default.
* **Decision:** **TODO.**
* **Reference:** `train_evaluate.py::fit_select`; `checks/test_frozen.py`.
* **Verification:** `pytest checks` - 32 passed (see `CHECKS.md`).

---

*Template for further records:*

```
## <ID> - <Design | Implementation | Checking>: <title>
* Task:
* Prompt:
* Response excerpt:
* Decision (accepted / corrected / rejected, and why):
* Code / change reference (file, commit):
* Verification executed and outcome:
```
