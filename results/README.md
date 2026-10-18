# Results

Output of the project's runs. Logs are unedited apart from removed progress-bar
lines, with two further exceptions: `step8_final.txt` and `step8_meanpool.txt`
were recovered from the executed notebook `final_model_colab_run.ipynb` (see
their headers), and the `lambda` column of `results_piece2.csv` was filled in
from each row's run tag after a logging bug. The main `README.md` maps each
table of the paper to its command and log.

Some logs were produced by earlier versions of the scripts. Those versions
printed working notes for the team (for example, suggested git commands and an
`owner` label for each run); the current scripts no longer print them, and the
numbers are unaffected.

| Files | Contents |
|---|---|
| `step*.txt` | the numbered scripts in `notebooks/`, in order |
| `results*.csv` | one row per seed for the corresponding experiments |
| `piece2_report.txt`, `piece3_report.txt`, `piece4_nosold_report.txt` | Table V |
| `piece4_report.txt` | an earlier distillation run that still included SemiSOLD tweets overlapping with SOLD (superseded by `piece4_nosold_report.txt`) |
| `extra_*.txt` | Table III |
| `error_analysis*.txt`, `error_analysis_raw*.json` | Table IV and Fig. 2; the JSON files hold every test prediction |
| `efficiency_minimal_*.txt`, `transformer_efficiency_*.txt` | Table VI |
| `*_run.ipynb` | executed runner notebooks, with their full output |
| `*_summary.txt` | the summary cells of those notebooks |
