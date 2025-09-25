import json

input_file = "/store/rsiva/code/navid_ws/NaVid-VLN-CE/tmp/gpt-dataset/training_data_merged.jsonl"
output_file = "/store/rsiva/code/navid_ws/NaVid-VLN-CE/tmp/gpt-dataset/300step_merged_training.jsonl"
num_samples = 300

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for i, line in enumerate(infile):
        if i >= num_samples:
            break
        # Optional: validate it's valid JSON
        data = json.loads(line)
        outfile.write(json.dumps(data) + "\n")

print(f"Saved first {num_samples} samples to {output_file}")
