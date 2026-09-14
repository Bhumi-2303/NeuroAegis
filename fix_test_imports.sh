find tests/reproducibility/ -name "*.py" -exec sed -i \
  -e 's|research\.experiments|research.experiments|g' {} +
