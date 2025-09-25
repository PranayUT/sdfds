import os
import argparse
import json
import math

def check_inf_nan(value):
    if math.isinf(value) or math.isnan(value):
        return 0
    return value

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    type=str,
    required=True,
)
args = parser.parse_args()

jsons = os.listdir(os.path.join(args.path, 'log'))
succ = 0
spl = 0
distance_to_goal = 0
path_length = 0
oracle_succ = 0

good_scenes = []

for j in jsons:
    with open(os.path.join(args.path, 'log', j)) as f:
        try:
            data = json.load(f)
            s = check_inf_nan(int(data['success']))
            sp = check_inf_nan(data['spl'])
            pl = check_inf_nan(data['path_length'])

            succ += s
            spl += sp
            distance_to_goal += check_inf_nan(data['distance_to_goal'])
            oracle_succ += check_inf_nan(int(data['oracle_success']))
            path_length += pl

            # collect "good" scenes: success + valid SPL
            if s == 1:
                good_scenes.append((j, sp, pl))

        except Exception as e:
            print(f"Error in {j}: {e}")

print(f'Success rate: {succ}/{len(jsons)} ({succ/len(jsons):.3f})')
print(f'Oracle success rate: {oracle_succ}/{len(jsons)} ({oracle_succ/len(jsons):.3f})')
print(f'SPL: {spl:.3f}/{len(jsons)} ({spl/len(jsons):.3f})')
print(f'Distance to goal: {distance_to_goal/len(jsons):.3f}')
print(f'Path length: {path_length/len(jsons):.3f}')

# sort scenes by SPL (desc), then path_length (desc)
good_scenes.sort(key=lambda x: (x[1], x[2]), reverse=True)

print("\nTop 20 good scenes (by SPL, then path length):")
for j, sp, pl in good_scenes[:20]:
    print(f"{j} (SPL={sp:.3f}, PathLen={pl:.3f})")
