#!/usr/bin/env python3
import os
import json
import pandas as pd
import argparse
import shutil

def csv_to_qwen_json(input_csv_path, output_json_path, expected_benchmark_name=None):
    """Convert a CSV (with label_xsmall/... etc) into the narrow qwen JSON format."""
    print(f"  Converting CSV {input_csv_path} → {output_json_path}...")
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"    ❌ Failed to read CSV: {e}")
        return False

    required_label_cols = ["label_xsmall", "label_small", "label_medium", "label_large"]
    if "doc_id" not in df.columns or "input_text" not in df.columns:
        print(f"    ❌ Missing required columns 'doc_id' or 'input_text' in {input_csv_path}")
        return False

    if expected_benchmark_name and "benchmark_name" in df.columns:
        # Filter to the expected benchmark if provided
        df = df[df["benchmark_name"].astype(str).str.lower() == expected_benchmark_name.lower()]
        if df.empty:
            print(f"    ❌ No rows matching benchmark_name='{expected_benchmark_name}' in {input_csv_path}")
            return False

    missing_labels = [c for c in required_label_cols if c not in df.columns]
    if missing_labels:
        print(f"    ❌ Missing label columns in CSV: {missing_labels}")
        return False

    # Build list of dicts with the qwen raw format (having label_* keys)
    records = []
    for _, row in df.iterrows():
        try:
            rec = {
                "doc_id": int(row["doc_id"]),
                "input_text": str(row["input_text"]),
            }
            # include label_* fields
            for size in ["xsmall", "small", "medium", "large"]:
                label_key = f"label_{size}"
                rec[label_key] = float(row[label_key]) if pd.notna(row[label_key]) else 0.0
            # Optionally include other fields if needed (e.g., acc_norm_*, energy_consumption_*, etc.)
            records.append(rec)
        except Exception as e:
            print(f"    Warning: skipping row due to error: {e}")
            continue

    if not records:
        print(f"    ❌ No valid records produced from {input_csv_path}")
        return False

    # Write JSON
    try:
        with open(output_json_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"    ✅ Wrote qwen JSON: {output_json_path} ({len(records)} entries)")
        return True
    except Exception as e:
        print(f"    ❌ Failed to write JSON: {e}")
        return False

def convert_json_format(input_json_path, output_json_path):
    """Convert JSON from qwen2_narrow_cost_spread format to inference_outputs format"""
    try:
        print(f"  Converting {input_json_path}...")
        
        # Load the input JSON
        with open(input_json_path, 'r') as f:
            input_data = json.load(f)
        
        print(f"    Loaded {len(input_data)} entries")
        
        # Convert to target format
        converted_data = []
        
        for entry in input_data:
            try:
                # Create converted entry
                converted_entry = {
                    "doc_id": int(entry.get("doc_id", 0)),
                    "question": f"Question: {entry.get('input_text', '')}",
                    "scores": {}
                }
                
                # Convert label_* fields to scores object
                for size in ['xsmall', 'small', 'medium', 'large']:
                    label_key = f"label_{size}"
                    if label_key in entry:
                        try:
                            score_value = float(entry[label_key]) if pd.notna(entry[label_key]) else 0.0
                            converted_entry["scores"][size] = score_value
                        except (ValueError, TypeError):
                            converted_entry["scores"][size] = 0.0
                    else:
                        converted_entry["scores"][size] = 0.0
                
                # Only add if we have a valid question
                if converted_entry["question"].strip() != "Question: ":
                    converted_data.append(converted_entry)
                    
            except Exception as e:
                print(f"    Error processing entry with doc_id {entry.get('doc_id', 'unknown')}: {e}")
                continue
        
        print(f"    Converted {len(converted_data)} valid entries")
        
        if not converted_data:
            print(f"    No valid data to save")
            return False
        
        # Create backup before overwriting (if replacing original file)
        if input_json_path == output_json_path:
            backup_path = input_json_path + ".backup"
            print(f"    Creating backup: {backup_path}")
            shutil.copy2(input_json_path, backup_path)
        
        # Save converted data (overwrites original if paths are the same)
        with open(output_json_path, 'w') as f:
            json.dump(converted_data, f, indent=2)
        
        print(f"    Saved converted JSON: {output_json_path}")
        
        # Show sample of converted data
        if converted_data:
            print(f"    Sample entry:")
            print(f"      Doc ID: {converted_data[0]['doc_id']}")
            print(f"      Question: {converted_data[0]['question'][:100]}...")
            print(f"      Scores: {converted_data[0]['scores']}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Error converting {input_json_path}: {e}")
        if input_json_path == output_json_path:
            backup_path = input_json_path + ".backup"
            if os.path.exists(backup_path):
                print(f"    Restoring backup due to error...")
                shutil.copy2(backup_path, input_json_path)
        return False

def process_folder(folder_path, central_csv_df=None):
    """Process a single folder: generate qwen JSON (from csv) if missing, then convert to inference_outputs format"""
    folder_name = os.path.basename(folder_path)
    print(f"\nProcessing folder: {folder_name}")
    
    json_filename = f"{folder_name}_qwen.json"
    json_path = os.path.join(folder_path, json_filename)
    
    # If JSON is missing, try to create it from CSV
    if not os.path.exists(json_path):
        # Priority 1: folder-local all_data.csv
        local_csv = os.path.join(folder_path, "all_data.csv")
        created = False
        if os.path.exists(local_csv):
            created = csv_to_qwen_json(local_csv, json_path, expected_benchmark_name=folder_name)
        elif central_csv_df is not None:
            # Attempt to extract subset for this benchmark
            subset = central_csv_df[central_csv_df["benchmark_name"].astype(str).str.lower() == folder_name.lower()]
            if not subset.empty:
                # Write subset to a temp CSV-like dict then to JSON
                tmp_json_records = []
                for _, row in subset.iterrows():
                    try:
                        rec = {
                            "doc_id": int(row["doc_id"]),
                            "input_text": str(row["input_text"]),
                        }
                        for size in ["xsmall", "small", "medium", "large"]:
                            label_key = f"label_{size}"
                            rec[label_key] = float(row[label_key]) if pd.notna(row[label_key]) else 0.0
                        tmp_json_records.append(rec)
                    except Exception as e:
                        print(f"    Warning: skipping row during central split: {e}")
                if tmp_json_records:
                    with open(json_path, "w") as f:
                        json.dump(tmp_json_records, f, indent=2)
                    print(f"    ✅ Created {json_filename} from central CSV ({len(tmp_json_records)} entries)")
                    created = True
        else:
            print(f"  No source CSV found to generate {json_filename}, skipping JSON creation.")
        
        if not created:
            print(f"  ❌ Could not create JSON for {folder_name}, skipping conversion.")
            return False  # skip further processing

    # At this point json_path should exist
    success = convert_json_format(json_path, json_path)
    if success:
        print(f"  ✅ Successfully processed {folder_name}")
    else:
        print(f"  ❌ Failed to process {folder_name}")
    return success

def main():
    parser = argparse.ArgumentParser(description="Combine/convert qwen2_mixed data (CSV → qwen JSON → inference_outputs JSON).")
    parser.add_argument("--base-path", "-b", default="data/qwen2_mixed", help="Base path containing benchmark subfolders.")
    parser.add_argument("--central-csv", "-c", default=None, help="Optional central all_data.csv to split by benchmark_name.")
    args = parser.parse_args()

    base_path = args.base_path
    central_csv_df = None

    print(f"Processing folders in: {base_path}")

    if not os.path.exists(base_path):
        print(f"❌ Base path does not exist: {base_path}")
        return

    # Load central CSV once if provided
    if args.central_csv:
        if os.path.exists(args.central_csv):
            try:
                central_csv_df = pd.read_csv(args.central_csv)
                if "benchmark_name" not in central_csv_df.columns:
                    print(f"❌ central CSV {args.central_csv} is missing 'benchmark_name' column; ignoring it.")
                    central_csv_df = None
                else:
                    print(f"Loaded central CSV: {args.central_csv} with {len(central_csv_df)} rows")
            except Exception as e:
                print(f"❌ Failed to load central CSV {args.central_csv}: {e}")
                central_csv_df = None
        else:
            print(f"❌ central CSV path does not exist: {args.central_csv}")

    # Get all subdirectories
    folders = [f for f in os.listdir(base_path)
               if os.path.isdir(os.path.join(base_path, f))]

    if not folders:
        print("No folders found to process")
        return

    print(f"Found {len(folders)} folders to process:")
    for folder in sorted(folders):
        print(f"  - {folder}")

    print("\n" + "="*60)
    print("Starting processing...")

    processed_count = 0
    converted_jsons = []
    for folder in sorted(folders):
        folder_path = os.path.join(base_path, folder)
        try:
            ok = process_folder(folder_path, central_csv_df=central_csv_df)
            if ok:
                processed_count += 1
                json_file = os.path.join(folder_path, f"{folder}_qwen.json")
                if os.path.exists(json_file):
                    converted_jsons.append(json_file)
        except Exception as e:
            print(f"❌ Error processing folder {folder}: {e}")
            continue

    print("\n" + "="*60)
    print("✅ Processing complete!")

    # Summary
    print(f"\nSummary:")
    print(f"  Processed folders: {processed_count}/{len(folders)}")
    print(f"  JSON files present: {len(converted_jsons)}")
    if converted_jsons:
        print(f"\nConverted JSON files:")
        for json_file in converted_jsons:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    print(f"  - {os.path.basename(json_file)}: {len(data)} entries")
            except Exception:
                print(f"  - {os.path.basename(json_file)}: Error reading file")

if __name__ == "__main__":
    main()
