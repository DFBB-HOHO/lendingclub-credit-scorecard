# Data dictionary and modelling use

| Field | Meaning | Use |
|---|---|---|
| `issue_d` | Loan issue month | Time split and vintage diagnostics only |
| `annual_inc` | Self-reported annual income | Candidate predictor |
| `dti` | Debt-to-income ratio excluding mortgage and requested loan | Candidate predictor |
| `loan_amnt` | Requested loan amount | Candidate predictor |
| `fico_n` | Mean of reported FICO range | Candidate predictor |
| `term` | Requested repayment term (36/60 months) | Candidate predictor |
| `revol_util`, `bc_util` | Revolving/bankcard utilisation | Candidate predictors |
| `inq_last_6mths` | Recent credit inquiries | Candidate predictor |
| `acc_open_past_24mths` | Recently opened trades | Candidate predictor |
| `credit_history_months` | Derived months since earliest credit line | Candidate predictor |
| `emp_length` | Employment-length band | Candidate predictor |
| `purpose` | Stated loan purpose | Candidate predictor |
| `home_ownership` | Home-ownership status | Candidate predictor |
| `Default` | 1 = charged off/default, 0 = fully paid | Target |
| `id` | Loan identifier | Duplicate check only; excluded from model |
| `addr_state`, `zip_code` | Geography | Excluded: proxy/fair-lending and stability risk |
| `title`, `desc` | Applicant text | Excluded: missingness, free-text instability and governance cost |

The pipeline deliberately removes LendingClub-assigned `grade`, `sub_grade`,
interest rate and all repayment/performance fields because they are outputs of
the platform's underwriting or arise after origination. Candidate bureau fields
were checked against the accompanying LendingClub data dictionary.
