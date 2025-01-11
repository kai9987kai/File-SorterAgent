import os
import shutil
import random
import numpy as np
import tensorflow as tf
from collections import deque
from typing import List, Tuple
import argparse
import datetime
from docx import Document
import asyncio
import csv

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

###############################################################################
# Global Configuration
###############################################################################
settings = {
    "agent": {
        "max_steps": 1000,  # Maximum steps per episode
        "energy": 100,      # Energy level (simulated)
        "energy_decay": 0.1,
        "energy_regeneration": 0.02,
    },
    "memory_size": 20000,
    "batch_size": 64,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "learning_rate": 0.0005,
    "reward": {
        "correct_move": 20,     # Increased reward for correct moves
        "incorrect_move": -20,  # Increased penalty for incorrect moves
        "no_move": -1,          # Small penalty for not moving a file
    },
}

# File Types and Corresponding Folders
file_types = {
    "documents": [".pdf", ".docx", ".txt", ".xlsx", ".pptx"],
    "images": [".jpg", ".png", ".jpeg", ".gif", ".bmp"],
    "videos": [".mp4", ".mkv", ".avi", ".mov"],
    "music": [".mp3", ".wav", ".flac", ".aac"],
    "archives": [".zip", ".rar", ".tar", ".gz"],
}

# Keywords for Dynamic Folder Naming
dynamic_folders = {
    "work": ["report", "meeting", "project"],
    "personal": ["holiday", "family", "travel"],
}

###############################################################################
# Helper Functions
###############################################################################
def get_file_type(file_name: str) -> str:
    """Determine the type of a file based on its extension."""
    _, ext = os.path.splitext(file_name)
    for folder, extensions in file_types.items():
        if ext.lower() in extensions:
            return folder
    return "other"

def read_text_file(file_path: str) -> str:
    """Read the content of a text file (txt or docx)."""
    try:
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return ""

def extract_keywords(content: str) -> List[str]:
    """Extract keywords from file content."""
    words = content.split()
    return [word.lower() for word in words if len(word) > 3]

def get_state(file_path: str) -> np.ndarray:
    """
    Create a state representation for the agent.
    Includes one-hot encoding for file type, size, creation year, 
    and presence of dynamic-folder keywords.
    """
    file_name = os.path.basename(file_path)
    file_type = get_file_type(file_name)

    # Size in MB
    file_size = os.path.getsize(file_path) / (1024 * 1024)  

    creation_date = os.path.getctime(file_path)
    creation_date = datetime.datetime.fromtimestamp(creation_date).strftime("%Y-%m-%d")

    # Read text files for additional context
    content = ""
    if file_type == "documents":
        content = read_text_file(file_path)
    keywords = extract_keywords(content)

    # One-hot encoding of file type (length = number of known folders + 1 for 'other')
    state = np.zeros(len(file_types) + 1, dtype=np.float32)
    folder_keys = list(file_types.keys())
    if file_type in folder_keys:
        state[folder_keys.index(file_type)] = 1.0
    else:
        state[-1] = 1.0  # "other" category

    # Add metadata to the state (file_size, creation_year)
    creation_year = float(creation_date.split("-")[0])  # year as float
    state = np.append(state, [file_size, creation_year])

    # Add keyword presence for dynamic folders
    for folder, keywords_list in dynamic_folders.items():
        found_keyword = any(keyword in keywords for keyword in keywords_list)
        state = np.append(state, [1 if found_keyword else 0])

    return state

async def move_file(source: str, destination: str) -> bool:
    """
    Move a file from source to destination asynchronously.
    Returns True if the move was successful, False otherwise.
    """
    try:
        # Ensure the destination folder exists
        destination_folder = os.path.dirname(destination)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            print(f"Created folder: {destination_folder}")

        # Perform the move in a background thread
        await asyncio.to_thread(shutil.move, source, destination)
        print(f"Moved: {source} -> {destination}")
        return True
    except Exception as e:
        print(f"Error moving file: {e}")
        return False

def log_move(source: str, destination: str, reward: float):
    """
    Log the file move (source -> destination) and the reward to a CSV file.
    """
    log_filename = "file_moves.log"
    try:
        with open(log_filename, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([datetime.datetime.now().isoformat(), source, destination, reward])
    except Exception as e:
        print(f"Failed to log file move: {e}")

###############################################################################
# Dueling DQN Model
###############################################################################
def build_dueling_dqn(input_shape: Tuple[int], num_actions: int) -> tf.keras.Model:
    """Build a Dueling DQN model."""
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)

    # Value stream
    value = tf.keras.layers.Dense(1, activation="linear")(x)

    # Advantage stream
    advantage = tf.keras.layers.Dense(num_actions, activation="linear")(x)

    # Combine value and advantage
    mean_advantage = tf.keras.layers.Lambda(lambda m: tf.reduce_mean(m, axis=1, keepdims=True))(advantage)
    q_values = tf.keras.layers.Add()([
        value, 
        tf.keras.layers.Subtract()([advantage, mean_advantage])
    ])

    model = tf.keras.Model(inputs=inputs, outputs=q_values)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=settings["learning_rate"]), 
        loss="mse"
    )
    return model

###############################################################################
# File Organizer Agent
###############################################################################
class FileOrganizerAgent:
    def __init__(self, num_actions: int):
        """
        Actions:
          0: Move to "documents"
          1: Move to "images"
          2: Move to "videos"
          3: Move to "music"
          4: Move to "archives"
          5: Do not move (leave in "other")
        """
        self.num_actions = num_actions
        self.energy = settings["agent"]["energy"]
        self.memory = deque(maxlen=settings["memory_size"])
        self.epsilon = settings["epsilon"]

        # Create Dueling DQN models
        input_len = len(file_types) + 1 + 2 + len(dynamic_folders)  # matches get_state(...) output length
        self.model = build_dueling_dqn((input_len,), num_actions)
        self.target_model = build_dueling_dqn((input_len,), num_actions)
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        q_values = self.model.predict(state[np.newaxis, ...], verbose=0)[0]
        return int(np.argmax(q_values))

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train the model using experiences from memory."""
        if len(self.memory) < settings["batch_size"]:
            return

        batch = random.sample(self.memory, settings["batch_size"])
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # Double DQN update
        next_q_local = self.model.predict(next_states, verbose=0)
        next_actions = np.argmax(next_q_local, axis=1)
        q_target_vals = self.target_model.predict(next_states, verbose=0)

        target_f = self.model.predict(states, verbose=0)
        for i in range(settings["batch_size"]):
            a = next_actions[i]
            if dones[i]:
                target = rewards[i]
            else:
                target = rewards[i] + settings["gamma"] * q_target_vals[i][a]
            target_f[i][actions[i]] = target

        self.model.fit(states, target_f, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > settings["epsilon_min"]:
            self.epsilon *= settings["epsilon_decay"]

    def update_target_model(self):
        """Update the target model with the current model's weights."""
        self.target_model.set_weights(self.model.get_weights())

    async def step(self, files: List[str], current_index: int) -> Tuple[bool, float]:
        """
        Perform one step of the simulation.
        Returns (alive, reward).
        """
        file_path = files[current_index]
        state = get_state(file_path)
        action = self.act(state)

        file_name = os.path.basename(file_path)
        file_type = get_file_type(file_name)

        # Determine destination folder from the action
        if action < len(file_types):
            destination_folder = list(file_types.keys())[action]
            destination = os.path.join(destination_folder, file_name)
        else:
            # Action is "do not move" => no actual move, but we treat it as "other"
            destination_folder = "other"
            destination = os.path.join(destination_folder, file_name)

        # Try to move the file if action is < len(file_types)
        if action < len(file_types):
            success = await move_file(file_path, destination)
        else:
            success = True  # No move physically done, but let's treat it as successful

        # Calculate reward
        if success:
            # If file was physically moved (or agent chose no move), check correctness
            if destination_folder == file_type:
                reward = settings["reward"]["correct_move"]
            else:
                # If the agent chose "do not move" but the file actually belongs somewhere else,
                # or if it moved it incorrectly
                reward = settings["reward"]["incorrect_move"]
        else:
            # Could not move the file, penalize slightly
            reward = settings["reward"]["no_move"]

        # Update agent's energy
        self.energy -= settings["agent"]["energy_decay"]
        if self.energy < 0:
            self.energy = 0

        # Check if episode is done
        done = (current_index >= len(files) - 1) or (self.energy <= 0)

        # Next state
        if not done and (current_index + 1 < len(files)):
            next_state = get_state(files[current_index + 1])
        else:
            next_state = state

        # Store experience
        self.remember(state, action, reward, next_state, done)

        # Train the agent
        self.replay()

        # --------------------------------------------------------------------
        # INNOVATION: Log every move attempt (even if success=False or "no move").
        # --------------------------------------------------------------------
        if action < len(file_types):
            # If physically moved, log source->destination
            log_move(file_path, destination, reward)
        else:
            # If no move, log source->"other" as the "destination"
            log_move(file_path, f"(no move) => {destination}", reward)

        # --------------------------------------------------------------------
        # IMPORTANT FIX: If the file was physically moved, remove it from the
        # list so we do NOT attempt get_state(...) on a non-existent path later.
        # --------------------------------------------------------------------
        if action < len(file_types):
            # If we tried to move physically...
            if success:
                files.pop(current_index)
            else:
                current_index += 1
        else:
            # No move was physically done; keep the file in place, move on
            current_index += 1

        return not done, reward

###############################################################################
# Commander Agent
###############################################################################
class CommanderAgent:
    def __init__(self, num_agents: int, input_shape: Tuple[int]):
        """
        Oversees communication between multiple agents 
        (in this script, we only have one, but it can be extended).
        """
        self.num_agents = num_agents
        self.input_shape = input_shape
        self.memory = deque(maxlen=100000)  # Larger memory for shared experiences
        self.epsilon = 1.0

        # Neural network for learning communication policies
        self.model = self.build_communication_model()
        self.target_model = self.build_communication_model()
        self.target_model.set_weights(self.model.get_weights())

    def build_communication_model(self) -> tf.keras.Model:
        """Builds a simple communication model."""
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Dense(128, activation="relu")(inputs)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        # Output: Probability distribution over num_agents
        outputs = tf.keras.layers.Dense(self.num_agents, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="mse")
        return model

    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy policy for deciding which agent to communicate with."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_agents)
        q_values = self.model.predict(state[np.newaxis, ...], verbose=0)[0]
        return int(np.argmax(q_values))

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train the model from shared experiences."""
        if len(self.memory) < 128:
            return
        batch = random.sample(self.memory, 128)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # Double DQN logic
        next_q_local = self.model.predict(next_states, verbose=0)
        next_actions = np.argmax(next_q_local, axis=1)
        q_target_vals = self.target_model.predict(next_states, verbose=0)

        target_f = self.model.predict(states, verbose=0)
        for i in range(128):
            a = next_actions[i]
            if dones[i]:
                target = rewards[i]
            else:
                target = rewards[i] + 0.99 * q_target_vals[i][a]
            target_f[i][actions[i]] = target

        self.model.fit(states, target_f, epochs=1, verbose=0)

        if self.epsilon > 0.05:
            self.epsilon *= 0.995

    def update_target_model(self):
        """Update the target model weights."""
        self.target_model.set_weights(self.model.get_weights())

    def communicate(self, agent_states: List[np.ndarray]) -> List[int]:
        """
        Decide which agents should share information based on their states.
        Returns a list of actions (which agent to share with).
        """
        shared_info = []
        for state in agent_states:
            action = self.act(state)
            shared_info.append(action)
        return shared_info

###############################################################################
# Main Simulation Loop with Commander Agent
###############################################################################
async def run_simulation_with_commander(directory: str):
    """
    Run the file organizer simulation with the Commander Agent.
    """
    script_name = os.path.basename(__file__)  # Exclude the script itself
    all_files = os.listdir(directory)
    files = [
        os.path.join(directory, f)
        for f in all_files
        if os.path.isfile(os.path.join(directory, f))
           and f != script_name
    ]

    if not files:
        print("No files to organize in the directory.")
        return

    # Create the File Organizer Agent
    file_organizer_agent = FileOrganizerAgent(num_actions=len(file_types) + 1)

    # CommanderAgent setup
    sample_state = get_state(files[0])
    input_shape = sample_state.shape
    commander_agent = CommanderAgent(num_agents=1, input_shape=input_shape)  # Only 1 agent for now

    total_reward = 0.0
    current_index = 0
    update_target_counter = 0

    try:
        while current_index < len(files):
            alive, reward = await file_organizer_agent.step(files, current_index)
            total_reward += reward

            # Communicate with Commander Agent
            # (In this simple example we have only one agent, but you can extend it.)
            state = get_state(files[current_index]) if current_index < len(files) else sample_state
            shared_info = commander_agent.communicate([state])
            print(f"Commander shared info: {shared_info}")

            if not alive:
                print("Episode ended (energy depleted or last file processed).")
                break

            # If the file was physically moved and popped from the list, 
            # the 'current_index' might still point to a valid next file 
            # or we might be at the end.
            if current_index >= len(files):
                break

            # Update target networks periodically
            update_target_counter += 1
            if update_target_counter % 100 == 0:
                file_organizer_agent.update_target_model()
                commander_agent.update_target_model()

            print(f"Step: {current_index}, Reward: {reward}, Total Reward: {total_reward:.2f}")

        # End of while loop
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")

    print("Simulation ended.")
    print(f"Total reward accumulated: {total_reward:.2f}")

###############################################################################
# Script Entry Point
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Commander Agent with File Organizer")
    parser.add_argument(
        "--directory",
        type=str,
        default=".",
        help="Directory to organize (default: current directory)"
    )
    args = parser.parse_args()
    directory_to_organize = args.directory
    asyncio.run(run_simulation_with_commander(directory_to_organize))
