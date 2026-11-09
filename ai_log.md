# AI development log

The brief (page 24) requires at least six records: two **design**, two **implementation**, two **checking/debugging**, each with task, prompt, response excerpt, decision, code/change reference and executed verification. At least two must describe a proposal that was **rejected or corrected**.

The records below are factual entries from the first development session (Claude, claude.ai, 8 Nov 2026). The *Decision* fields marked **TODO** must be completed by the team in your own words after you have reviewed the code. Add further records from your own later sessions; keep only records you can stand behind.

---

## D1 - Design: choosing Option B and the module layout

* **Task:** choose between Project A and Project B and plan the code organisation.
* **Prompt (summary):** shared the course page and asked to start the final project.
* **Response excerpt:** compared both briefs (data, methods, screens, LLM dependency) and recommended B as "fewer moving parts, no API dependency, closer to the lecture material"; proposed `data.py` / `features.py` / `models.py` / `train_evaluate.py` / `app.py` / `checks/`.
* **Decision:** chose Option B, mainly because of resources: Project A depends on a paid LLM agent at run time and is more complicated to build and run, while Project B can be built and reproduced with standard Python libraries only.
* **Reference:** repository layout, first commit.
* **Verification:** layout matches brief page 22.

## D2 - Design: historical inputs by exact timestamp lookup

* **Task:** decide how lag features are built so that missing hours cannot leak or shift.
* **Prompt (summary):** implement the page 6 inputs with the page 4 audit rules.
* **Response excerpt:** used `clean[col].reindex(t - k hours)` (exact lookup) instead of `shift(k)`, plus a rolling 24-hour mean over the *complete* hourly calendar with `min_periods=24`.
* **Decision:** accepted. The inputs must not misrepresent the data to fit what we need: if an hour is absent, using the previous row would silently give a different hour's count (for 2011-01-02 06:00, the 04:00 count instead of the missing 05:00). Marking the input as missing and flagging it is the honest treatment.
* **Reference:** `features.py` (`build_raw_features`, `build_raw_features_fast`).
* **Verification:** `checks/test_features.py::test_missing_previous_timestamp_is_missing_not_shifted`, `test_real_missing_hour_2011_01_02_05`, `test_24h_mean_requires_complete_window`, `test_brief_worked_example_inputs` all pass (the 2012-08-01 08:00 example reproduces 487, 760, 788, 300.4167, 0.64, 0.83, 0.1343).

## I1 - Implementation: data source (correction)

* **Task:** download `hour.csv`.
* **Response excerpt:** the first attempt to download from archive.ics.uci.edu was refused by the sandbox network; the assistant fetched the file from a GitHub copy instead.
* **Correction / decision:** a mirror is not automatically trustworthy, so the file was verified before use. We did not rely on that check alone: the team downloaded `hour.csv` from the official UCI page and compared it with the repository copy. The two files are byte-for-byte identical (same SHA-256, `e03de4ee...504f`, as listed in the README; 17,379 rows, same columns and values), so the repository data is the original UCI file.
* **Reference:** `data/hour.csv`, README "Data".
* **Verification:** 17,379 rows; values quoted in the brief (2012-08-01 07:00 = 487, 08:00 = 681; 2011-01-02 05:00 absent) checked by script and by `checks/test_frozen.py::test_audit_finds_brief_missing_hour`.

## I2 - Implementation: penalty grids (correction)

* **Task:** fit Ridge, Lasso, Poisson and Bayesian candidates and select by validation MAE.
* **Response excerpt (first version):** grids Ridge {0.1 ... 1000}, Lasso {0.01 ... 30}, Poisson {1e-5 ... 0.1}, Bayesian alpha {1e-6 ... 1}.
* **Problem found:** every method selected the *smallest* value in its grid, so the optimum might lie outside the grid.
* **Correction:** grids extended downwards (Ridge to 0.001, Lasso to 0.001, Poisson to 1e-7, Bayesian alpha to 1e-8). Validation MAE then flattens (Ridge 50.228 at 0.001 and 0.01), showing a plateau rather than a missed optimum.
* **Decision:** accepted. We agreed with the assistant that a selected value at the edge of a grid means the search may be incomplete, so extending the grid was necessary. Because validation MAE stayed practically the same after the extension, we conclude that these models need very little regularisation with 11,539 training hours and 59 inputs, and that the chosen settings are not an artefact of the original grid.
* **Reference:** `train_evaluate.py` `GRIDS`; `outputs/validation_results.csv`.
* **Verification:** rerun of `train_evaluate.py`; Regularisation panel in the app.

## C1 - Checking: Poisson errors much larger than the linear models

* **Task:** explain Poisson validation MAE 84.0 vs about 50.2 for Ridge.
* **Response excerpt:** checked convergence (`n_iter_` = 236 of 5000, converged) and inspected the largest forecasts: evening peaks such as 2012-09-11 18:00 forecast 2531 against 877 recorded. With a log link, the standardised lag counts act multiplicatively, so the higher 2012 demand is extrapolated exponentially.
* **Decision:** kept the model, because the brief requires the same transformed input vector for all regression methods; documented it as a failure mode with a proposed change (log-transformed lag counts for the Poisson model, tested on validation). We reviewed the explanation and agreed: changing the inputs for one model only would break the like-for-like comparison the brief asks for, so the weakness is reported rather than hidden.
* **Reference:** `models.py::PoissonModel`; report "Limitations".
* **Verification:** saved predictions inspected; Poisson coverage 23.5% at nominal 80% recorded in `metrics.csv`.

## C2 - Checking: proving learned values do not depend on test outcomes

* **Task:** implement the page 23 check that changing test values cannot alter preprocessing, coefficients or the default.
* **Response excerpt:** refactored steps 2-4 of `train_evaluate.py` into `fit_select(clean)`, then wrote a check that reruns it on data whose test-period `cnt` and `temp` were replaced by random values and asserts identical preprocessing, coefficients, validation table and default.
* **Decision:** accepted. We agreed with the assistant's reasoning: the test scores are only an honest estimate of future performance if nothing learned (medians, scaling, coefficients, chosen penalties, default model) could have been influenced by test outcomes. Reading the code is not enough to be sure of this; the check demonstrates it by running the fit on deliberately corrupted test data.
* **Reference:** `train_evaluate.py::fit_select`; `checks/test_frozen.py`.
* **Verification:** `pytest checks` - 32 passed (see `CHECKS.md`).

## C3 - Checking: Lasso penalty selector drawn wrongly after the tab reorder (correction)

*From a later Claude Code session, 8 Nov 2026, during the dashboard redesign.*

* **Task:** check the restyled Inspect screen in a real browser. The redesign moved the Regularisation tab from first to fourth position, so it is hidden when the screen loads.
* **Prompt (summary):** no prompt was about this control. The redesign request said "Start the app, look at every screen at desktop width and at ~400px width, fix overflow"; the assistant found the problem during that check.
* **Response excerpt (translated from Hebrew):** "the penalty slider shows a wrong value after being moved to a hidden tab: the handle is at the right end with the value 0.001. I will replace it with a selectbox." More exactly, the label 0.001 was correct but the handle was drawn at the right end, although 0.001 is the first of the ten sorted options. The overflow script also flagged an `INPUT` extending past the viewport (right edge 1454 at 1440 px wide, 488 at 400 px). The hidden tab is the likely cause, but it was not isolated: the slider was not tried in a visible tab.
* **Decision:** **TODO.**
* **Reference:** `app.py::tab_regularisation`, `st.select_slider` replaced by `st.selectbox("Penalty for coefficient table", ...)`, default still the selected penalty; commit `607031b`.
* **Verification:** after the change the overflow script no longer flagged the `INPUT` at either width, and the 400 px screenshot shows the selector at 0.001. A one-off Streamlit `AppTest` run (not part of `checks/`) showed intercept 148.033 at the default and 151.166 after selecting penalty 1.0, both equal to `outputs/lasso_coefficients.csv`, with no exception. `python -m pytest checks`: 32 passed. Changing the selection was not exercised by hand in the browser.

## I3 - Implementation: the screen recording was produced by the AI assistant (disclosure)

*From the claude.ai session, 9 Nov 2026.*

* **Task:** produce the ~3-minute screen recording required by the brief (page 25): forecast and reveal, inputs table with source times, model comparison, one error inspection, a CSV download.
* **Prompt (summary):** the team asked the assistant to make the recording itself, with on-screen captions, and then asked for a narration script suitable for text-to-speech (numbers, times and abbreviations written out in words) plus suggestions for a closing line.
* **Response excerpt:** the assistant ran the dashboard (commit `ce4726f`) in a headless Chromium browser, drove it with a Playwright script (clicks, scrolling and a drawn mouse pointer), captured the browser frames, burned in English captions with ffmpeg, and added a title card and an end card. Captions were written from the values the app displays at each step (e.g. 664.4 forecast, 681 recorded, MAE 48.96, -57.7% vs baseline, largest error 2012-10-02 08:00, 80% coverage 72.3%).
* **Decision:** the video content and captions are AI-produced. The narration is computer-generated: the team added it in Clipchamp with its text-to-speech feature, using the assistant's narration script, and placed each line on the timeline itself. We used the captions and narration script without changes.
* **Reference:** published video https://youtu.be/gACUjhpPY7w (unlisted), linked from `README.md`. Source files `bike_dashboard_demo.mp4` and `captions.srt` are not stored in the repository.
* **Verification:** the assistant inspected still frames at each step before and after rendering (screens, caption placement, revealed values). The recorded values match `outputs/predictions.csv` and `outputs/metrics.csv`. The team watched the final narrated video in full before publishing it.

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
