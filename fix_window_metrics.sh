find . -type f -name "*.py" -not -path "*/\.*" -not -path "*/venv/*" -not -path "*/node_modules/*" -exec sed -i \
  -e 's/research\.experiments\.imbalance\.metrics/neuroaegis.evaluation.window_metrics/g' \
  {} +
