import argparse
import json
from neuroaegis.tracking.config import diff_configs

def main():
    parser = argparse.ArgumentParser(description="Diff two experiment configs")
    parser.add_argument("config1", type=str, help="Path to first config yaml")
    parser.add_argument("config2", type=str, help="Path to second config yaml")
    args = parser.parse_args()
    
    diffs = diff_configs(args.config1, args.config2)
    
    if not diffs:
        print("Configs are identical.")
    else:
        print("Config Differences (config1 -> config2):")
        for k, (v1, v2) in diffs.items():
            print(f"  {k}: {v1} -> {v2}")

if __name__ == "__main__":
    main()
