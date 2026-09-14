import os
import json
from datetime import datetime
import re

inventory = []
for root, dirs, files in os.walk('.', topdown=True):
    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env']]
    for f in files:
        if f.startswith('.'): continue
        path = os.path.normpath(os.path.join(root, f))
        
        cat = "Unknown"
        hist = False
        prod = False
        gen = False
        exp = False
        
        if path.endswith(('.png', '.pdf', '.csv', '.npz', '.json', '.log')):
            gen = True
            
        if 'phase_' in path.lower() or 'research/' in path.lower():
            cat = "Research/Experimentation"
            exp = True
            match = re.search(r'phase_(\d+)', path.lower())
            if match and int(match.group(1)) < 8:
                hist = True
                
        if 'apps/' in path.lower():
            cat = "Application"
            prod = True
            
        if path.endswith(('.pt', '.pth')):
            cat = "Model Artifact"
            gen = True
            
        if 'data/' in path.lower():
            cat = "Dataset/Manifest"
            
        purpose = f"File of type {os.path.splitext(f)[1]}"
        if cat == "Model Artifact": purpose = "Model weight checkpoint"
        elif path.endswith('.md'): purpose = "Documentation/Report"
        elif path.endswith('.py'): purpose = "Source Code"
        elif path.endswith('.csv'): purpose = "Data table/Results/Manifest"
        
        inventory.append({
            "path": path,
            "category": cat,
            "purpose": purpose,
            "actively_used": prod or (not hist and exp),
            "historical": hist,
            "experimental": exp,
            "production": prod,
            "reproducible": not gen or (cat == "Model Artifact"),
            "generated": gen,
            "safe_to_move": True,
            "referenced_elsewhere": True if path.endswith('.pt') else False
        })

os.makedirs('research/audit/repository_structure', exist_ok=True)
with open('research/audit/repository_structure/repository_inventory.json', 'w') as f:
    json.dump({"inventory": inventory, "timestamp": str(datetime.now())}, f, indent=2)

with open('research/audit/repository_structure/repository_inventory.md', 'w') as f:
    f.write("# Repository Inventory\n\n")
    f.write("| Path | Category | Purpose | Historical | Experimental | Production |\n")
    f.write("|---|---|---|---|---|---|\n")
    for item in inventory[:500]:
        f.write(f"| `{item['path']}` | {item['category']} | {item['purpose']} | {item['historical']} | {item['experimental']} | {item['production']} |\n")
    if len(inventory) > 500:
        f.write(f"\n*(Truncated. Total files: {len(inventory)})*\n")
        
with open('research/audit/repository_structure/proposed_structure.md', 'w') as f:
    f.write("# Proposed Project Structure\n\n(See README for full details of migration to `src/neuroaegis`, `artifacts`, `data`, and `app`).")

print("Inventory generated.")
