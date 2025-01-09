import os
import shutil
import random
import numpy as np
import tensorflow as tf
from collections import deque
from typing import List, Tuple
import argparse

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings

# Global Configuration
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
        "correct_move": 20,  # Increased reward for correct moves
        "incorrect_move": -20,  # Increased penalty for incorrect moves
        "no_move": -1,  # Small penalty for not moving a file
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

# Helper Functions
def get_file_type(file_name: str) -> str:
    """Determine the type of a file based on its extension."""
    _, ext = os.path.splitext(file_name)
    for folder, extensions in file_types.items():
        if ext.lower() in extensions:
            return folder
    return "other"

def get_state(files: List[str], current_index: int) -> np.ndarray:
    """Create a state representation for the agent."""
    # One-hot encoding of the current file type
    file_type = get_file_type(files[current_index])
    state = np.zeros(len(file_types) + 1, dtype=np.float32)  # +1 for "other"
    if file_type in file_types:
        state[list(file_types.keys()).index(file_type)] = 1.0
    else:
        state[-1] = 1.0  # "other" category
    return state

def move_file(source: str, destination: str) -> bool:
    """Move a file from source to destination."""
    try:
        # Ensure the destination folder exists
        destination_folder = os.path.dirname(destination)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            print(f"Created folder: {destination_folder}")

        shutil.move(source, destination)
        print(f"Moved: {source} -> {destination}")  # Log the move
        return True
    except Exception as e:
        print(f"Error moving file: {e}")
        return False

# Dueling DQN Model
def build_dueling_dqn(input_shape: Tuple[int], num_actions: int) -> tf.keras.Model:
    """Build a Dueling DQN model."""
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)

    # Value stream
    value = tf.keras.layers.Dense(1, activation="linear")(x)

    # Advantage stream
    advantage = tf.keras.layers.Dense(num_actions, activation="linear")(x)

    # Combine value and advantage using Keras layers
    mean_advantage = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantage)
    q_values = tf.keras.layers.Add()([value, tf.keras.layers.Subtract()([advantage, mean_advantage])])

    model = tf.keras.Model(inputs=inputs, outputs=q_values)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=settings["learning_rate"]), loss="mse")
    return model

# Agent Class
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
        self.model = build_dueling_dqn((len(file_types) + 1,), num_actions)
        self.target_model = build_dueling_dqn((len(file_types) + 1,), num_actions)
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        q_values = self.model.predict(state[np.newaxis, ...], verbose=0)[0]
        return np.argmax(q_values)

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

    def step(self, files: List[str], current_index: int) -> Tuple[bool, float]:
        """Perform one step of the simulation."""
        state = get_state(files, current_index)
        action = self.act(state)

        file_name = files[current_index]
        file_type = get_file_type(file_name)
        destination_folder = list(file_types.keys())[action] if action < len(file_types) else "other"

        # Move the file
        if action < len(file_types):
            destination = os.path.join(destination_folder, file_name)
            success = move_file(file_name, destination)
        else:
            success = True  # No move

        # Calculate reward
        if success:
            if destination_folder == file_type:
                reward = settings["reward"]["correct_move"]
            else:
                reward = settings["reward"]["incorrect_move"]
        else:
            reward = settings["reward"]["no_move"]

        # Update energy
        self.energy -= settings["agent"]["energy_decay"]
        if self.energy < 0:
            self.energy = 0

        # Check if episode is done
        done = (current_index >= len(files) - 1) or (self.energy <= 0)

        # Store experience
        next_state = get_state(files, current_index + 1) if not done else state
        self.remember(state, action, reward, next_state, done)

        # Train the agent
        self.replay()

        return not done, reward

# Main Simulation Loop
def run_simulation(directory: str):
    """Run the file organizer simulation."""
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    agent = FileOrganizerAgent(num_actions=len(file_types) + 1)

    total_reward = 0.0
    current_index = 0
    update_target_counter = 0

    try:
        while current_index < len(files):
            alive, reward = agent.step(files, current_index)
            total_reward += reward
            if not alive:
                print("Episode ended.")
                break

            current_index += 1

            # Update target network periodically
            update_target_counter += 1
            if update_target_counter % 100 == 0:
                agent.update_target_model()

            print(f"Step: {current_index}, Reward: {reward}, Total Reward: {total_reward}")

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")

    print("Simulation ended.")
    print(f"Total reward accumulated: {total_reward:.2f}")

# Script Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="File Organizer Agent")
    parser.add_argument(
        "--directory",
        type=str,
        default=".",
        help="Directory to organize (default: current directory)",
    )
    args = parser.parse_args()
    directory_to_organize = args.directory
    run_simulation(directory_to_organize)
