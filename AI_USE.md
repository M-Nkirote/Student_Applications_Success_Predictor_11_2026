# AI use disclosure

## Tool used
Claude, used through the claude.ai chat interface.

## What the AI assisted with
- **Review of the notebook results:** reviewed the executed notebook outputs and suggested improvements, including the leakage checks, the training window for right-censoring, the multiple-enrolment rule, feature selection and the final model-choice rule.
- **Writing code functions that I didn't know how to, but knew what they were meant to do**. 

## What I did myself
- Chose the problem and dataset and defined the business context (application stages, what counts as success).
- Supplied domain knowledge that changed modelling decisions, for example that deals are never reassigned (so `Deal Owner` is a legitimate feature), when `Study Destination Type` and `Visa Is Required` are filled in, and that spam records had to be removed.
- Ran every notebook version and the pipeline on the real data, checked the outputs, and decided which suggestions to accept.
- Code writing and changes 
- Presentation creation

## How AI output was verified
- All code was executed on the real dataset; the scripted pipeline was checked to reproduce the notebook's results.
- Automated tests (`pytest`) cover cleaning rules, feature engineering, the API and the SHAP additivity property.
- Modelling claims in the report are based on the executed outputs, not on AI statements.
