find . -type f -name "*.py" -not -path "*/\.*" -not -path "*/venv/*" -not -path "*/node_modules/*" -exec sed -i \
  -e 's/neuroaegis\.models\.baselines\.cnn_gnn_gru_model/neuroaegis.models.baselines.model_c/g' \
  {} +
