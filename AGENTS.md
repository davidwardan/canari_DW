# AGENTS.md
 
Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.
 
**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.
 
## 1. Think Before Coding
 
**Don't assume. Don't hide confusion. Surface tradeoffs.**
 
Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.
 
## 2. Simplicity First
 
**Minimum code that solves the problem. Nothing speculative.**
 
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
 
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.
 
## 3. Surgical Changes
 
**Touch only what you must. Clean up only your own mess.**
 
When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.
 
When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.
 
The test: Every changed line should trace directly to the user's request.
 
## 4. Goal-Driven Execution
 
**Define success criteria. Loop until verified.**
 
Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"
 
For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```
 
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
 
---
 
**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
 
## 5. Plotting
* Use the Matplotlib defaults below:

```python
import matplotlib as mpl

# Figure widths matching typical two-column paper layouts
SINGLE_COL = (3.5, 2.5)  # ~88mm — fits one column (default)
DOUBLE_COL = (6.5, 3.5)  # ~165mm — spans both columns

mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}",
    "lines.linewidth": 1,
    "figure.figsize": SINGLE_COL,
    "font.size": 9,
    "savefig.dpi": 300,
})
```

* Keep all plots clear, minimal, and publication-ready.
* Use `SINGLE_COL` by default and `DOUBLE_COL` only when necessary.
* Do not add figure titles.
* Label axes clearly and always include units when applicable.
* Avoid unnecessary gridlines, annotations, markers, colors, and visual clutter.
* Keep the number of plotted curves to the minimum needed to communicate the result.
* Use consistent colors and styles across experiments.
* Place legends above the plot or on the right, with no border.
* Prefer direct comparison within one figure only when curves remain easy to distinguish; otherwise use separate figures.
* Ensure text, ticks, and legends remain readable at the final printed figure size.
* Save every final figure as both `.pgf` and `.pdf`.

For time-series plots:

* observations: `tab:red`
* predictions: `tab:blue`
* epistemic uncertainty: `tab:blue`, `alpha=0.3`
* aleatoric uncertainty: `tab:green`, `alpha=0.3`

Use the project Matplotlib defaults and save figures under the experiment's `figures/` directory.

## 6. Experiments

* Write all experiments under `experiments/`, with one subfolder per experiment:

  ```
  experiments/exp_title/
  ├── data/
  ├── figures/
  ├── planning/
  ├── results/
  └── scripts/
  ```

* In `planning/README.md`, briefly define:

  * the research question and hypothesis;
  * dataset and train/validation/test split;
  * baselines and metrics;
  * expected outcome.

* Keep experiment-specific code outside `src/`. Modify `src/` only when necessary and document why.

* Make experiments reproducible:

  * set and record random seeds;
  * use multiple seeds when training is stochastic;
  * save configurations and raw per-run results;
  * do not overwrite previous runs.

* Prevent data leakage:

  * never use test data for training, tuning, early stopping, normalization, calibration, or threshold selection;
  * use identical splits and preprocessing when comparing methods.

* For probabilistic models, evaluate both prediction accuracy and uncertainty quality. Use metrics such as RMSE/MAE together with log-likelihood, CRPS.

* When separating epistemic and aleatoric uncertainty, clearly define how each is computed and verify that the decomposition is mathematically consistent.

* Before running the full experiment, test the pipeline on a simple toy problem where the expected behavior is known.

* Generate final tables and figures programmatically from saved results rather than manually copying values.

## 7. Testing

* Run relevant tests after meaningful code changes.
* Prefer focused tests during development.
* For numerical algorithms, test against analytically known or controlled cases.
* Add regression tests when fixing bugs.
* Do not delete or weaken tests simply to make them pass.
* Verify both expected behavior and important edge cases.

