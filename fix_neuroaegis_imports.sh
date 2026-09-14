find . -type f -name "*.py" -not -path "*/\.*" -not -path "*/venv/*" -not -path "*/node_modules/*" -exec sed -i \
  -e 's/research\.experiments\.gnn\.cnn_gnn_model/neuroaegis.models.baselines.cnn_gnn_model/g' \
  -e 's/research\.experiments\.model_c\.cnn_gnn_gru_model/neuroaegis.models.baselines.cnn_gnn_gru_model/g' \
  {} +
