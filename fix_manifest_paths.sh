find . -type f \( -name "*.py" -o -name "*.json" -o -name "*.md" \) -not -path "*/\.*" -not -path "*/venv/*" -not -path "*/node_modules/*" -exec sed -i \
  -e 's|research/data/manifests|data/manifests|g' \
  {} +
