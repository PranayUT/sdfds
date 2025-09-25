from openai import OpenAI
from dotenv import load_dotenv
import json
import numpy as np
from habitat import Env
from habitat.core.agent import Agent
from tqdm import trange
import os
import re
import cv2
import imageio
from habitat.utils.visualizations import maps
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
import random
import io
import base64
import matplotlib.pyplot as plt
from PIL import Image


load_dotenv()  # Load environment variables from a .env file if present
client = OpenAI()

# ft_job_id = "ftjob-RDxNBpSIN818ep2v8T6bA4bf"
# response = client.fine_tuning.jobs.retrieve(ft_job_id)
method1 = "ft:gpt-4.1-2025-04-14:ut-autonomous::CEURhSrA"
method2 = "ft:gpt-4.1-2025-04-14:ut-autonomous::CEVk7wfn"
method3 = "ft:gpt-4.1-2025-04-14:ut-autonomous::CEV9xiq1"
fine_tuned_model = method1
print(f"Fine-tuned model: {fine_tuned_model}")



def evaluate_agent(config, split_id, dataset, model_path, result_path) -> None:
    env = Env(config.TASK_CONFIG, dataset)

    # for testing
    result_path = "tmp/testing_tuned_1500"
    num_episodes = 2

    agent = MyGPTAgent(model_path, result_path)
    #num_episodes = len(env.episodes)
    EARLY_STOP_ROTATION = config.EVAL.EARLY_STOP_ROTATION
    EARLY_STOP_STEPS = config.EVAL.EARLY_STOP_STEPS

    target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success"}

    count = 0

    for _ in trange(
        num_episodes, desc=config.EVAL.IDENTIFICATION + "-{}".format(split_id)
    ):
        obs = env.reset()
        iter_step = 0
        agent.reset()

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

            if (
                continuse_rotation_count > EARLY_STOP_ROTATION
                or iter_step > EARLY_STOP_STEPS
            ):
                action = {"action": 0}

            iter_step += 1
            obs = env.step(action)

        info = env.get_metrics()
        result_dict = dict()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["id"] = env.current_episode.episode_id
        count += 1

        with open(
            os.path.join(
                os.path.join(result_path, "log"),
                "stats_{}.json".format(env.current_episode.episode_id),
            ),
            "w",
        ) as f:
            json.dump(result_dict, f, indent=4)


class MyGPTAgent(Agent):
    def __init__(self, model_path, result_path, require_map=True):
        # print("Initialize MyGPTAgent")

        self.result_path = result_path
        self.require_map = require_map

        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)

        # Initialize OpenAI client
        from dotenv import load_dotenv

        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize tracking variables
        self.rgb_list = []
        self.topdown_map_list = []
        self.count_id = 0
        self.previous_plan = None
        self.previous_output = None
        self.step_count = 0
        self.history = []
        self.history_window = 5
        self.last_action = None

        # print("Initialization Complete")

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
        Return 16-point cardinal label and *clockwise* yaw in degrees
        (0° = North, 90° = East, 180° = South, 270° = West).
        """
        heading_vector = quaternion_rotate_vector(
            quaternion.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        yaw = np.rad2deg(np.array(phi) + z_neg_z_flip)

        dirs = [
            "S",
            "SSE",
            "SE",
            "ESE",
            "E",
            "ENE",
            "NE",
            "NNE",
            "N",
            "NNW",
            "NW",
            "WNW",
            "W",
            "WSW",
            "SW",
            "SSW",
        ]
        idx = round(yaw / 24)
        return dirs[idx], yaw

    def parse_action_number(self, response_text):
        action_match = re.search(r"Action:\s*\[?(\d+)\]?", response_text, re.IGNORECASE)
        if action_match:
            action_num = int(action_match.group(1))
            if 0 <= action_num <= 3:
                return action_num

        try:
            first_char = response_text.strip()[0]
            if first_char.isdigit():
                action_num = int(first_char)
                if 0 <= action_num <= 3:
                    return action_num
        except (ValueError, IndexError):
            pass

        first_line = response_text.split("\n")[0]
        for char in first_line:
            if char in "0123":
                action_num = int(char)
                return action_num

        print("No valid action found, defaulting to 0 (stop)")
        return 0

    def parse_next_step(self, generated_text: str) -> str:
        match = re.search(
            r"Next step:\s*(.+)", generated_text, re.IGNORECASE | re.DOTALL
        )
        if match:
            next_step = match.group(1).strip()
            next_step = next_step.split("\n")[0].strip()
            return next_step
        else:
            return ""

    def get_topdown_map_base64(self, info, rgb_shape):
        if "top_down_map_vlnce" in info:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                info["top_down_map_vlnce"], rgb_shape[0]
            )

            if top_down_map.dtype != np.uint8:
                top_down_map = (top_down_map * 255).astype(np.uint8)

            plt.figure(figsize=(8, 8))
            plt.imshow(top_down_map)
            plt.title("Top-Down Map")
            plt.axis("off")
            buf = io.BytesIO()
            # plt.savefig(
            #     "tmp/testing/top_down_map.png",
            #     format="png",
            #     bbox_inches="tight",
            #     pad_inches=0,
            # )
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)

            img_bytes = buf.getvalue()
            map_img_str = base64.b64encode(img_bytes).decode("utf-8")

            plt.close()

            return map_img_str
        return None

    def get_simple_topdown_map_base64(self, env):
        agent_state = env.sim.get_agent_state()
        goal_pos = env.current_episode.goals[0].position
        top_down_map = maps.get_topdown_map_from_sim(env.sim)
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        top_down_map = recolor_map[top_down_map]
        # Convert from Habitat's coordinate system to matplotlib's
        coords = maps.to_grid(
            agent_state.position[2],
            agent_state.position[0],
            (top_down_map.shape[0], top_down_map.shape[1]),
            sim=env.sim,
        )
        # Convert from quaternion to yaw angle
        rot = agent_state.rotation
        yaw = np.pi + np.arctan2(
            2 * rot.y * rot.w - 2 * rot.x * rot.z,
            1 - 2 * rot.y * rot.y - 2 * rot.z * rot.z,
        )
        # Add marker for start agent position
        agent_map = maps.draw_agent(
            image=top_down_map,
            agent_center_coord=coords,
            agent_rotation=yaw,
            agent_radius_px=50,
        )

        plt.imshow(agent_map)
        plt.title("Annotated Top-Down Map")
        plt.grid(False)
        plt.axis("off")
        plt.savefig(
            "tmp/testing/top_down_map.png",
            format="png",
            bbox_inches="tight",
            pad_inches=0,
        )
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        goal_plt_pos = list(
            maps.to_grid(
                goal_pos[2],
                goal_pos[0],
                (top_down_map.shape[0], top_down_map.shape[1]),
                sim=env.sim,
            )
        )
        plt.scatter(goal_plt_pos[1], goal_plt_pos[0], c="yellow", s=30)
        plt.text(
            goal_plt_pos[1],
            goal_plt_pos[0] - 60,
            "Goal",
            fontsize=7,
            color="black",
            ha="center",
            va="top",
        )
        img_bytes = buf.getvalue()
        map_img_str = base64.b64encode(img_bytes).decode("utf-8")

        return map_img_str


    def addtext(self, image, instruction, navigation, current_direction, yaw):
        """Add text overlay to image with wrapping and shrink-to-fit"""
        h, w = image.shape[:2]
        new_height = h + 200
        new_image = np.ones((new_height, w, 3), np.uint8) * 255
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        line_spacing = 5

        # Build text
        all_text = f"Current direction: {current_direction}. {instruction}"
        lines = []
        words = all_text.split(" ")
        line = ""

        for word in words:
            test_line = (line + " " + word).strip()
            size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
            if size[0] > w - 20:  # wrap if too wide
                lines.append(line)
                line = word
            else:
                line = test_line
        if line:
            lines.append(line)

        # Add navigation on a new line
        if navigation == None:
            navigation = "No navigation reasoning provided."
        nav_words = navigation.split(" ")
        nav_line = ""
        for word in nav_words:
            test_line = (nav_line + " " + word).strip()
            size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
            if size[0] > w - 20:
                lines.append(nav_line)
                nav_line = word
            else:
                nav_line = test_line
        if nav_line:
            lines.append(nav_line)

        # Adjust font size if still overflowing vertically
        while True:
            text_height = len(lines) * (
                cv2.getTextSize("Test", font, font_scale, thickness)[0][1] + line_spacing
            )
            if h + text_height < new_height:
                break
            font_scale -= 0.05
            if font_scale < 0.3:
                break

        # Render text
        y = h + 30
        for line in lines:
            cv2.putText(new_image, line, (10, y), font, font_scale, (0, 0, 0), thickness)
            y += int(
                cv2.getTextSize(line, font, font_scale, thickness)[0][1] + line_spacing
            )

        return new_image

    def reset(self):
        if self.require_map:
            if len(self.topdown_map_list) != 0:
                output_video_path = os.path.join(
                    self.result_path, "video", "{}.gif".format(self.episode_id)
                )
                imageio.mimsave(output_video_path, self.topdown_map_list)

        self.rgb_list = []
        self.topdown_map_list = []
        self.previous_plan = None
        self.previous_output = None
        self.step_count = 0
        self.count_id += 1
        self.pending_action_list = []

    def act(self, observations, info, episode_id, env):
        self.episode_id = episode_id
        self.step_count += 1
        rgb = observations["rgb"]
        self.rgb_list.append(rgb)
        agent_state = env.sim.get_agent_state()
        if agent_state is not None:
            current_direction, current_yaw = self.get_cardinal_direction(
                agent_state.rotation
            )
        else:
            raise ValueError("Agent state not available")

        if (
            len(self.pending_action_list) != 0
        ):  # Pending action queue so GPT isn't queried every step
            temp_action = self.pending_action_list.pop(
                0
            )  # Run steps in queue before requerying gpt

            # if self.require_map:
            #     top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            #         info["top_down_map_vlnce"], rgb.shape[0]
            #     )
            #     output_im = np.concatenate((rgb, top_down_map), axis=1)
            #     img = self.addtext(
            #         output_im,
            #         observations["instruction"]["text"],
            #         f"Pending action: {temp_action}",
            #         current_direction,
            #         current_yaw
            #     )
            #     self.topdown_map_list.append(img)

            return {"action": temp_action}

        if self.step_count % 1 == 0:
            instruction = observations["instruction"]["text"]

            # collision = observations.get("collisions", {}).get("is_collision", False)
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
            self.history = self.history[-self.history_window :]

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

            map_img_str = self.get_topdown_map_base64(info, rgb.shape)
            image_data = self.encode_image(rgb)
            # map_img_str = self.get_topdown_map_base64(info, rgb.shape)
            user_text = (
                f"Navigate to approach the red square on the top down map using the top-down map and camera view.\n\n"
                f"TASK INSTRUCTION: Navigate until the agent's arrow is on top of the red square on the top down map.\n\n"
                f"AGENT ORIENTATION:\n"
                f"- Current cardinal direction: {current_direction}\n"
                f"- Yaw angle: {current_yaw:.1f}°\n\n"
            )

            distance_to_goal = info.get("distance_to_goal", None)
            if distance_to_goal is not None:
                user_text += f"\nDISTANCE TO GOAL: {distance_to_goal:.2f} meters\n\n"

            if self.previous_plan:
                user_text += f"Plans from previous step:\n{self.previous_plan}\n\n"

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

            messages = [  # System prompt for navigation
                {
                    "role": "system",
                    "content": (
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
                        # "=== INSTRUCTION HANDLING PRINCIPLES ==="
                        # "- Align map reasoning with the goal (if the instruction says 'enter the room on the left,' prioritize detecting a doorway on the left side of the map/camera).\n"
                        # "- If the map doesn’t directly show the described feature (e.g., 'hallway'), rely on camera view and relative navigation until a landmark matches.\n"
                        "=== HISTORY INTERPRETATION RULES ==="
                        "- If Collision=True, treat it as a collision (forward failed) → reroute using a different action.\n"
                        "- If the last few actions are all turns (2 or 3) and orientation is nearly unchanged, you are looping → choose a different strategy (try forward or opposite turn).\n"
                        "- Use history and past path to avoid repeating the same failed action sequence.\n"
                        "=== OUTPUT FORMAT ===\n"
                        "Action: [0-3]\n"
                        "Map reasoning: [Describe goal location relative to you in cardinal terms]\n"
                        "Camera reasoning: [Objects / obstacles seen in current view]\n"
                        "Navigation reasoning: [Step-by-step plan using map + camera]\n"
                        "Next step: [Brief plan for next move]"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{map_img_str}"
                            },
                        },
                    ],
                },
            ]
            try:
                response = self.client.chat.completions.create(
                    model=fine_tuned_model,
                    messages=messages,
                    max_tokens=300,
                    temperature=0.3,
                )



                generated_text = response.choices[0].message.content.strip()

                self.previous_plan = self.parse_next_step(generated_text)
                self.previous_output = generated_text
                action_index = self.parse_action_number(generated_text)
                # print(f"\nModel decision: {action_index}\n")

                if action_index == 0:
                    self.pending_action_list.append(0)
                elif action_index == 1:
                    for _ in range(
                        3
                    ):  # We add multiple low level actions to prevent GPT calls every step
                        self.pending_action_list.append(1)
                elif action_index == 2:
                    for _ in range(2):
                        self.pending_action_list.append(2)
                elif action_index == 3:
                    for _ in range(2):
                        self.pending_action_list.append(3)

            except Exception as e:
                print(f"API Error: {e}")
                self.pending_action_list.append(random.randint(1, 3))
        else:
            # self.pending_action_list.append(1)
            pass

        if len(self.pending_action_list) == 0:  # Start with a stop sig
            self.pending_action_list.append(0)

        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                info["top_down_map_vlnce"], rgb.shape[0]
            )
            output_im = np.concatenate((rgb, top_down_map), axis=1)

            action_text = f"Next action: {self.pending_action_list[0]}"
            if hasattr(self, "previous_plan") and self.previous_plan:
                action_text += f" | Plan: {self.previous_plan[:50]}..."

            img = self.addtext(
                output_im,
                observations["instruction"]["text"],
                self.previous_output,
                current_direction,
                current_yaw,
            )
            self.topdown_map_list.append(img)

        self.last_action = self.pending_action_list[0]

        return {"action": self.pending_action_list.pop(0)}