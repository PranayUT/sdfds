import json
import os
import glob
from pathlib import Path
import base64
from typing import Dict, List, Any
import pandas as pd
from io import BytesIO
from PIL import Image

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')  # Convert to RGB
        buffered = BytesIO()
        image.save(buffered, format="PNG") 
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

# def encode_image_to_base64(image_path: str) -> str:
#     """Convert image to base64 string for OpenAI format"""
#     try:
#         with open(image_path, "rb") as image_file:
#             return base64.b64encode(image_file.read()).decode('utf-8')
#     except Exception as e:
#         print(f"Error encoding image {image_path}: {e}")
#         return ""

def create_system_prompt() -> str:
    """Create the system prompt from the provided code"""
    return (
        "You are an AI navigation agent inside a simulated environment. "
        "Your job is to move from your START location to a GOAL location using your top-down map and camera view.\n\n"
        "=== MAP LEGEND ===\n"
        "- BLUE SQUARE: Your starting position.\n"
        "- BLUE ARROW: Your current position & facing direction.\n"
        "- BLUE LINE: Your path so far.\n"
        "- RED SQUARE: The goal location you must reach.\n"
        "- GRAY AREAS: Navigable floor where you can walk.\n"
        "- WHITE AREAS: Obstacles or walls you cannot walk through.\n\n"
        "=== NAVIGATION PRINCIPLES ===\n"
        "1. Always identify the RED SQUARE (goal) on the map.\n"
        "2. Compare your CURRENT CARDINAL DIRECTION (N, NE, E, SE, S, SW, W, NW) with the DIRECTION from your location to the goal.\n"
        "3. If not facing toward the goal, turn left (2) or right (3) to align your heading.\n"
        "4. Move forward (1) only when facing an open navigable path toward the goal using the FRONT VIEW CAMERA.\n"
        "5. Avoid white (non-navigable) areas — if blocked, reorient using the map.\n"
        "6. Stop (0) when you reach the goal on the top-down map (ONLY when blue arrow is touching or on top of red square).\n\n"
        "7. Use distance-to-goal as feedback: if it's close to zero, you are near the goal, and you should Stop(0).\n"
        "If it increases or stays constant for several steps, adjust strategy.\n\n"
        "=== DECISION RULES ===\n"
        "- Use top-down map to determine direction to move in.\n"
        "- Every step, make a micro-plan: identify goal direction, check navigability, choose turn/move.\n"
        "- If goal is to your left/right on the map, rotate toward it before moving.\n"
        "- Use global cardinal directions for reasoning, NOT relative left/right from the camera.\n\n"
        "=== HISTORY INTERPRETATION RULES ===\n"
        "- If Collision=True, treat it as a collision (forward failed) → reroute using a different action.\n"
        "- If the last few actions are all turns (2 or 3) and orientation is nearly unchanged, you are looping → choose a different strategy (try forward or opposite turn).\n"
        "- Use history and past path to avoid repeating the same failed action sequence.\n"
        "=== OUTPUT FORMAT ===\n"
        "Action: [0-3]\n"
    )

def create_user_text(current_direction: str, current_yaw: float, distance_to_goal: float, history_text: str) -> str:
    """Create the user text from the provided code template"""
    user_text = (
        f"Navigate to approach the red square on the top down map using the top-down map and camera view.\n\n"
        f"TASK INSTRUCTION: Navigate until the agent's arrow is on top of the red square on the top down map.\n\n"
        f"AGENT ORIENTATION:\n"
        f"- Current cardinal direction: {current_direction}\n"
        f"- Yaw angle: {current_yaw:.1f}°\n\n"
    )

    if distance_to_goal is not None:
        user_text += f"\nDISTANCE TO GOAL: {distance_to_goal:.2f} meters\n\n"

    if history_text:
        user_text += "=== HISTORY CONTEXT ===\n" + history_text + "\n"

    # Output options for actions
    user_text += (
        f"AVAILABLE ACTIONS:\n"
        f"0) Stop (task complete)\n"
        f"1) Move forward\n"
        f"2) Turn left\n"
        f"3) Turn right\n\n"
        f"Analyze the image and plan your next move using **global cardinal directions**."
    )
    
    return user_text

def process_episode(episode_file: str, base_dir: str) -> List[Dict[str, Any]]:
    """Process a single episode JSON file and return OpenAI format entries"""
    training_examples = []
    
    try:
        with open(episode_file, 'r') as f:
            episode_data = json.load(f)
        
        episode_id = episode_data['episode_id']
        steps = episode_data['steps']
        
        print(f"Processing episode {episode_id} with {len(steps)} steps...")
        
        for step_data in steps:
            # Extract step information
            action = step_data['action']
            rgb_path = os.path.join(base_dir, step_data['rgb_path'])
            map_path = os.path.join(base_dir, step_data['map_path'])
            cardinal_direction = step_data['cardinal_direction']
            yaw_deg = step_data['yaw_deg']
            distance_to_goal = step_data['distance_to_goal']
            history_text = step_data.get('history_text', '')
            
            # Check if image files exist
            if not os.path.exists(rgb_path):
                print(f"Warning: RGB image not found: {rgb_path}")
                continue
            if not os.path.exists(map_path):
                print(f"Warning: Map image not found: {map_path}")
                continue
            
            # Encode images to base64
            rgb_b64 = encode_image_to_base64(rgb_path)
            map_b64 = encode_image_to_base64(map_path)
            
            if not rgb_b64 or not map_b64:
                print(f"Warning: Failed to encode images for episode {episode_id}, step {step_data['step']}")
                continue
            
            # Create user text
            user_text = create_user_text(cardinal_direction, yaw_deg, distance_to_goal, history_text)
            
            # Create OpenAI format entry
            training_example = {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": create_system_prompt()}]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{rgb_b64}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{map_b64}"
                            }
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Action: "+str(action)}]
                }
            ]
        }

            
            training_examples.append(training_example)
            
    except Exception as e:
        print(f"Error processing episode file {episode_file}: {e}")
    
    return training_examples

def main():
    # Configuration
    json_dir = "/store/rsiva/code/navid_ws/NaVid-VLN-CE/tmp/gpt-dataset/json"
    base_dir = "/store/rsiva/code/navid_ws/NaVid-VLN-CE"  # Base directory for resolving image paths
    output_file = "/store/rsiva/code/navid_ws/NaVid-VLN-CE/tmp/gpt-dataset/navigation_training_data.jsonl"
    
    # Find all episode JSON files
    episode_files = glob.glob(os.path.join(json_dir, "episode_*.json"))
    episode_files.sort()  # Sort to process in order
    
    print(f"Found {len(episode_files)} episode files")
    
    # Process all episodes
    all_training_examples = []
    
    for episode_file in episode_files:
        training_examples = process_episode(episode_file, base_dir)
        all_training_examples.extend(training_examples)
        print(f"Processed {len(training_examples)} training examples from {os.path.basename(episode_file)}")
    
    # Write to JSONL file
    print(f"\nWriting {len(all_training_examples)} total training examples to {output_file}")
    
    with open(output_file, 'w') as f:
        for example in all_training_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Conversion complete! Output saved to {output_file}")
    print(f"Total training examples: {len(all_training_examples)}")

if __name__ == "__main__":
    main()