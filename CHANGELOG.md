# List of changes made to repository in response to feedback

All notable changes to this project are documented in this file.

This changelog focuses on improvements made in response to peer and TA feedback, as required by the assignment rubric.

The format follows guidance from [Keep a Changelog](https://keepachangelog.com/)

## [Unreleased]

### Fixed

**1. In commit [#7730c3d](https://github.com/clr-saunders/poisonous-mushroom-classifier/commit/7730c3dffd52f121b6bd134f046a32082846f5ce), addressed the following [feedback](https://github.com/UBC-MDS/data-analysis-review-2025/issues/14#issuecomment-3648581399):

"The model pipeline does not display well on the HTML report; therefore, it might be better to describe it using words or simply remove it since it is less relevant to the general audience."

**2. Removed ambiguous index column ("Unnamed: 0") from generated tables**

**Feedback addressed:**

>"Remove or rename the index column labeled 'Unnamed: 0'."
>"You should give a meaningful name to the index column of table 3 and remove 'Unnamed: 0' or make it an empty, invisible string."

**What was changed:**

- Refactored EDA utility functions to explicitly name index columns ("#") instead of relying on default pandas index serialization.
- Ensured all tables written to CSV and rendered in Quarto use a meaningful index label or no index where appropriate.

**Evidence:** Commit [#2340806](https://github.com/clr-saunders/poisonous-mushroom-classifier/commit/2340806734093e3568a2a89260a82d7305914154)
- Updated summarize_features() in src/mushroom_eda_utils.py to set a named index instead of leaving it unnamed.
- Updated compute_poison_variance_rank() output to avoid unnamed index columns.
- Verified rendered tables in docs/poisonous_mushroom_classifier.qmd no longer display Unnamed: 0.

**3. Removed large blocks of commented-out code from EDA scripts**

Feedback addressed:
>"I noticed in your scripts on a few occasions large blocks of code were commented out (for example in eda.py)."

**What was changed:**
- Removed obsolete and commented-out experimental code in scripts/eda.py that was no longer part of the final analysis pipeline.
- Consolidated logic into tested utility functions in `src/mushroom_eda_utils.py`.

**Evidence:** Commit [#fa93274](https://github.com/clr-saunders/poisonous-mushroom-classifier/pull/68/commits/fa93274b8f07456ce835d1e01d5c1f65b02eeac2)
- Cleaned `scripts/eda.py` to contain only executable pipeline logic.
- Moved reusable EDA logic into `src/mushroom_eda_utils.py`.

**4. Added attribution and corrected link in Code of Conduct**

**Feedback adreesed:**
>"Include attribution for the Code of Conduct and ensure that the source link is working.”

**What was changed:**

- Added proper attribution to the Contributor Covenant in the Code of Conduct document.
- Fixed the broken hyperlink to the official version (https://www.contributor-covenant.org/version/3/0/code_of_conduct/).
- Verified that the rendered link functions correctly in the HTML report and GitHub view.

**Evidence:** Commits [#8b2482f](https://github.com/clr-saunders/poisonous-mushroom-classifier/commit/8b2482fd93c819fd612fe70a61a02b0acc67a950) and [#ba63d79](https://github.com/clr-saunders/poisonous-mushroom-classifier/commit/ba63d79d0e5eeaed9a5a8eddc6f40a1c953c11d7).
- File: `CODE_OF_CONDUCT.md`

**5. Corrected minor typos and improved phrasing in project report**

**Feedback addressed:**

> "Some sections of the report contain small grammatical errors or awkward phrasing that affect readability."

**What was changed:**

- Reviewed and edited `docs/poisonous_mushroom_classifier.qmd` for grammatical consistency and clarity.  
- Revised the "Data Splitting" paragraph in `docs/poisonous_mushroom_classifier.qmd` for smoother, more formal phrasing.
- Improved the sentence about setting `random_state=123` to ensure reproducibility.
- Corrected minor typographical errors, punctuation inconsistencies, and formatting issues.  

**Evidence:** Commit [#5876bfc](https://github.com/clr-saunders/poisonous-mushroom-classifier/commit/5876bfc62d70f7696576acdcca25bb579cbfaef9)  
- File: `docs/poisonous_mushroom_classifier.qmd`
---

### Added

**4. Automated tests for EDA utility functions**

**Feedback addressed:**
>"Tests: Are there automated tests or manual steps described so that the function of the software can be verified?

**What was changed:**
- Added a dedicated test suite for EDA utilities using pytest.
- Tests cover:
    - Feature matrix construction
    - Feature summarization
    - Poison-rate computation
    - Cramér’s V calculation
    -Feature ranking by poison variance

**Evidence:** Commits [#90353c2](https://github.com/clr-saunders/poisonous-mushroom-classifier/pull/68/commits/90353c251bf2e9c9ab96bc1a8f44d7687ced61d3), [#3380d74](https://github.com/clr-saunders/poisonous-mushroom-classifier/pull/68/commits/3380d74747675a9c42bcc67dc19fda23a281fb9c) and [#fa93274](https://github.com/clr-saunders/poisonous-mushroom-classifier/pull/68/commits/fa93274b8f07456ce835d1e01d5c1f65b02eeac2)
- New test file: `tests/test_mushroom_eda_utils.py`
- All tests pass locally and inside Docker using `python -m pytest -q`
- Tests include both typical and edge cases (e.g., constant features, perfect association).

---

### Improved

**5. Expanded and standardized docstrings across scripts for clarity and reviewability**

**Feedback addressed:**
>"The docstrings in some of the script files could be more thorough to aid the project review or future user. For example, in the download_data.py script the docstring for the main function is only a one line general description… In Milestone 4 when we add function tests you could also add a ‘Raises’ section…"

**What was changed:**
- Expanded docstrings across multiple scripts to include:
    - Clear descriptions of input parameters
    - Return values and/or side effects (e.g., file creation)
    - Explicit Raises sections documenting how erroneous input is handled
    - Where appropriate, brief example usage snippets
- Refactored EDA-related logic into src/mushroom_eda_utils.py with fully documented, testable functions.
- Updated docstrings in EDA utilities and pipeline scripts to be consistent and reviewer-friendly.

**Evidence:** Commit [#3e4d9e4](https://github.com/clr-saunders/poisonous-mushroom-classifier/pull/68/commits/3e4d9e4aa495a2c05dfe0e9c1ba96384b8a054a0)
Expanded docstrings in:
- `src/mushroom_eda_utils.py` (e.g., get_poison_rate_by, stacked_poison_chart, compute_poison_variance_rank)
- `scripts/eda.py`
- Docstrings now follow a structured format including Parameters, Returns, Raises, and Examples where relevant.
- Tests in tests/test_mushroom_eda_utils.py align with documented error handling behavior.


**6. Expanded docstring and removed obsolete commented-out code in `download_data.py`**

**Feedback addressed:**

> "The docstrings in some of the script files could be more thorough to aid project review or future users"

**What was changed:**

- Rewrote the `main()` function docstring in `scripts/download_data.py` using a structured NumPy-style format.  
- Added detailed sections for "Parameters*", "Returns" and "Raises" to improve clarity and testability.  
- Removed outdated commented out code that was no longer relevant to the final pipeline.  


**Evidence:** Commit [#4d85360](https://github.com/clr-saunders/poisonous-mushroom-classifier/commit/4d853600ccd9bebe78d270849971645a9bef0e24)  
- File: `scripts/download_data.py`

---

### Notes
- All changes were made to improve clarity, robustness, and reproducibility, without altering the analytical conclusions of the project.
- Some feedback overlapped with TA guidance; in those cases, changes are documented once but address both sources.
