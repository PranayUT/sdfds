#!/usr/bin/env python3
import json
import numpy as np
import os
import cv2
import imageio
import io
import base64
import random
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import trange

from habitat import Env
from habitat.core.agent import Agent
from habitat.utils.visualizations import maps
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower



def evaluate_agent(config, split_id, dataset, model_path, result_path) -> None:
    """
    Runs a few episodes using the Shortest Path Follower expert policy,
    logs metrics, and (optionally) saves dataset samples and GIFs.
    """
    env = Env(config.TASK_CONFIG, dataset)

    result_path = result_path or "tmp/gpt-dataset"
    num_episodes = 10

    agent = ShortestPathAgent(result_path=result_path,
                              require_map=True,
                              save_dataset=True,       
                              dataset_dir="tmp/gpt-dataset",
                              frames_per_forward=1)    # repeat forward actions if we want to use less steps

    EARLY_STOP_ROTATION = config.EVAL.EARLY_STOP_ROTATION
    EARLY_STOP_STEPS = config.EVAL.EARLY_STOP_STEPS

    target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success"}

    for _ in trange(num_episodes, desc=f"{config.EVAL.IDENTIFICATION}-{split_id}"):
        obs = env.reset()
        iter_step = 0
        agent.reset()  # resets video buffers etc.

        # Attach env on first episode (needed to init SPF with sim)
        agent.attach_env(env)

        continuse_rotation_count = 0
        last_dtg = 999

        while not env.episode_over:
            info = env.get_metrics()

            if info["distance_to_goal"] != last_dtg:
                last_dtg = info["distance_to_goal"]
                continuse_rotation_count = 0
            else:
                continuse_rotation_count += 1

            action = agent.act(obs, info, env.current_episode.episode_id, env)

            # Early stopping fallbacks (mirror your original logic)
            if (continuse_rotation_count > EARLY_STOP_ROTATION or
                iter_step > EARLY_STOP_STEPS):
                action = {"action": 0}  # STOP

            iter_step += 1
            obs = env.step(action)

        # Episode done: dump metrics
        info = env.get_metrics()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["id"] = env.current_episode.episode_id

        os.makedirs(os.path.join(result_path, "log"), exist_ok=True)
        with open(
            os.path.join(result_path, "log", f"stats_{env.current_episode.episode_id}.json"),
            "w",
        ) as f:
            json.dump(result_dict, f, indent=4)



class ShortestPathAgent(Agent):
    """
    Expert agent that queries Habitat's ShortestPathFollower at every step
    and (optionally) logs a dataset of (RGB, top-down map, action).
    Also writes to a JSONL file for OpenAI multimodal finetuning.
    """
    def __init__(self,
                 result_path="tmp/testing",
                 require_map=True,
                 save_dataset=True,
                 dataset_dir="tmp/testing/dataset",
                 frames_per_forward=1):
        self.result_path = result_path
        self.require_map = require_map
        self.save_dataset = save_dataset
        self.dataset_dir = dataset_dir
        self.frames_per_forward = frames_per_forward

        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)
        if self.save_dataset:
            os.makedirs(self.dataset_dir, exist_ok=True)
            os.makedirs(os.path.join(self.dataset_dir, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_dir, "map"), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_dir, "json"), exist_ok=True)

        # Path for JSONL file
        self.jsonl_path = os.path.join(self.dataset_dir, "dataset.jsonl")
        if os.path.exists(self.jsonl_path):
            os.remove(self.jsonl_path)

        # Buffers / trackers
        self.rgb_list = []
        self.topdown_map_list = []
        self.count_id = 0
        self.step_count = 0
        self.last_action = None
        
        # NEW: Episode-level data collection
        self.current_episode_data = []
        self.current_episode_id = None

        # SPF handle + env
        self.follower = None
        self.env = None

        self.history = []

        self.reset()

    def encode_image(self, image_array):
        buffered = io.BytesIO()
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)

        if len(image_array.shape) == 3:
            if image_array.shape[2] == 4:
                img = Image.fromarray(image_array, mode="RGBA")
            else:
                img = Image.fromarray(image_array, mode="RGB")
        else:
            img = Image.fromarray(image_array)

        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_cardinal_direction(self, quaternion):
        """
        Return 16-point cardinal label and clockwise yaw in degrees
        (0° = North, 90° = East, 180° = South, 270° = West).
        """
        heading_vector = quaternion_rotate_vector(
            quaternion.inverse(), np.array([0, 0, -1])
        )
        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        yaw = np.rad2deg(np.array(phi) + z_neg_z_flip)
        dirs = [
            "S","SSE","SE","ESE","E","ENE","NE","NNE",
            "N","NNW","NW","WNW","W","WSW","SW","SSW",
        ]
        idx = round(yaw / 24)
        return dirs[idx], float(yaw)

    def addtext(self, image, instruction, navigation, current_direction, yaw):
        """Overlay text block under an image (kept from your version)."""
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instruction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY
        all_text = f"Current direction: {current_direction} (yaw {yaw:.1f}°).\n" + instruction
        words = all_text.split(" ")
        x = 10
        line = ""

        for word in words:
            test_line = line + " " + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)
            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1] + 5
            else:
                line = test_line

        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)

        y_line = y_line + textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)
        return new_image

    def get_topdown_map_frame(self, info, rgb_shape):
        """Colorize Habitat top-down map to align height with the RGB frame for side-by-side logging."""
        if "top_down_map_vlnce" not in info:
            return None
        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map_vlnce"], rgb_shape[0]
        )
        if top_down_map.dtype != np.uint8:
            top_down_map = (top_down_map * 255).astype(np.uint8)
        return top_down_map

    def attach_env(self, env):
        """Call once per episode (or before first act) so we can init SPF with the simulator."""
        self.env = env
        # (Re)create follower bound to the current sim
        self.follower = ShortestPathFollower(self.env.sim, goal_radius=0.2, return_one_hot=False)

    def reset(self):
        """End-of-episode cleanup + new-episode init."""
        # NEW: Save episode JSON if we have data
        if self.save_dataset and self.current_episode_data and self.current_episode_id is not None:
            episode_json_path = os.path.join(self.dataset_dir, "json", f"episode_{self.current_episode_id}.json")
            episode_summary = {
                "episode_id": str(self.current_episode_id),
                "total_steps": len(self.current_episode_data),
                "steps": self.current_episode_data
            }
            with open(episode_json_path, "w") as jf:
                json.dump(episode_summary, jf, indent=2)
            print(f"Saved episode data to {episode_json_path}")
        
        # save GIF (top-down visualization)
        if self.require_map and len(self.topdown_map_list) != 0:
            output_video_path = os.path.join(self.result_path, "video", f"{getattr(self, 'episode_id', 'ep')}.gif")
            imageio.mimsave(output_video_path, self.topdown_map_list)

        # reset buffers
        self.rgb_list = []
        self.topdown_map_list = []
        self.step_count = 0
        self.count_id += 1
        
        # NEW: Reset episode data
        self.current_episode_data = []
        self.current_episode_id = None

    def act(self, observations, info, episode_id, env):
        """
        Query SPF for the next action based on the CURRENT state and goal.
        Also logs (rgb, top-down, action) if save_dataset=True.
        Also appends to dataset.jsonl for finetuning.
        """
        self.episode_id = episode_id
        self.step_count += 1
        
        # NEW: Track current episode
        if self.current_episode_id != episode_id:
            self.current_episode_id = episode_id

        # Ensure follower is bound to this env/sim
        if self.follower is None or self.env is None or self.env is not env:
            self.attach_env(env)

        # Current first-person frame
        rgb = observations["rgb"]

        # Orientation info
        agent_state = env.sim.get_agent_state()
        current_direction, current_yaw = self.get_cardinal_direction(agent_state.rotation)

    
        collision_info = info.get(
                "collisions", 0
            )  # Added a collision measurement for agent's decision making
        collision = (
                collision_info.get("is_collision", False)
                if isinstance(collision_info, dict)
                else False
            )  # returns T or F if in collision

        self.history.append(
                {
                    "step": self.step_count,
                    "action": getattr(self, "last_action", None),
                    "direction": current_direction,
                    "yaw": current_yaw,
                    "collision": collision,
                }
            )
        self.history = self.history[-5 :]

        history_text = "Recent actions (last 5):\n"  # Summarize last 5 actions for better decision making
        for h in self.history:
            if h["action"] is None:
                continue
            hist_line = (
                    f"Step {h['step']}: Action={h['action']} | "
                    f"Dir={h['direction']} | Yaw={h['yaw']:.1f} | "
                    f"Collision={h['collision']}"
                )
            history_text += hist_line + "\n"

        if len(self.history) >= 4:
            last_actions = [
                    h["action"] for h in self.history if h["action"] is not None
                ]
            if len(last_actions) >= 4:
                if sum(a in [2, 3] for a in last_actions) == len(last_actions):
                    history_text += "\n⚠️ Warning: Loop detected (rotating in place).\n"  # If repeated turns, agent is stuck in loop

        distance_to_goal = info.get("distance_to_goal", -1.0)  # NEW: Distance to goal

        # Top-down frame
        map_frame = self.get_topdown_map_frame(info, rgb.shape)
        if map_frame is not None:
            output_im = np.concatenate((rgb, map_frame), axis=1)
            action_text = "Expert: shortest-path follower"
            instruction_text = observations.get("instruction", {}).get("text", "")
            vis = self.addtext(output_im, instruction_text, action_text, current_direction, current_yaw)
            if self.require_map:
                self.topdown_map_list.append(vis)

        goal_position = env.current_episode.goals[0].position
        action = self.follower.get_next_action(goal_position)
        if action is None:
            action = 0  # STOP if already within success radius

        if self.save_dataset:
            ep = str(episode_id)
            step = self.step_count

            # Save RGB image
            rgb_path = os.path.join(self.dataset_dir, "rgb", f"rgb_{ep}_{step:05d}.png")
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            # Save map image (if present)
            map_path = None
            if map_frame is not None:
                map_path = os.path.join(self.dataset_dir, "map", f"map_{ep}_{step:05d}.png")
                cv2.imwrite(map_path, cv2.cvtColor(map_frame, cv2.COLOR_RGB2BGR))

            # NEW: Add step data to episode collection instead of saving individual JSON
            step_data = {
                "step": step,
                "action": int(action),
                "rgb_path": rgb_path,
                "map_path": map_path,
                "cardinal_direction": current_direction,
                "yaw_deg": current_yaw,
                "distance_to_goal": distance_to_goal,  # NEW: Distance to goal
                "history_text": history_text,  # NEW: History text in exact GPT agent format
            }
            self.current_episode_data.append(step_data)

        self.last_action = int(action)
        print(f"[SPF] ep={episode_id} step={self.step_count} action={self.last_action}")
        return {"action": self.last_action}