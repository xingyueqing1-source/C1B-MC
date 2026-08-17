# C1B-MC
Official code for Conformalized 1-Bit Matrix Completion (C1B-MC). A distribution-free framework using weighted conformal prediction to provide finite-sample valid uncertainty quantification for binary matrices, robust to severe model misspecification.
# Project Structure
- `src/`: Core implementation of the C1B-MC algorithm (estimators, inference, data generation).
- `experiments/`: Scripts to reproduce all synthetic and real-world experiments reported in the paper.
- `visualization/`: Scripts to generate all figures and plots.
- `data/`: Contains the MovieLens-100k dataset used for real-world evaluation.
# Requirements
To install the required dependencies, run:
```bash
pip install -r requirements.txt
