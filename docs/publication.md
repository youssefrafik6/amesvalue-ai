# Publication handoff

Target owner and name are `youssefrafik6/amesvalue-ai`. The GitHub repository description is “End-to-end real estate price prediction and analytics platform featuring statistical regression, explainable machine learning, prediction intervals, and interactive dashboards.”

The local repository includes aggregate results, reproducible code, genuine dashboard captures and the fitted serving model. It excludes the original Excel workbook, course PDF and observation-level data products. The documentation links to the original publisher. The course report may be linked or added after its redistribution is authorized.

After public publication has been approved and the destination name is confirmed available, run these commands from the repository root.

```bash
gh api user --jq .login
gh repo create youssefrafik6/amesvalue-ai --public --source=. --remote=origin --push \
  --description 'End-to-end real estate price prediction and analytics platform featuring statistical regression, explainable machine learning, prediction intervals, and interactive dashboards.'
gh repo view youssefrafik6/amesvalue-ai --json nameWithOwner,url,visibility,description
```
