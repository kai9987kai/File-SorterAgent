# File-SorterAgent

The File Organizer Agent is a reinforcement learning-based system designed to automatically organize files in a directory. It uses a neural network to decide which folder a file should be moved to (e.g., documents, images, music) based on its type. The agent learns through trial and error, receiving rewards for correct moves and penalties for mistakes. Over time, it improves its accuracy by training on past experiences, making it an efficient and adaptive solution for file organization.

### Instructions for Using the File Organizer Agent

Follow these steps to use the **File Organizer Agent** to automatically organize files in a directory:

---

### 1. **Prerequisites**
- **Python 3.x** installed on your system.
- Required Python libraries: `numpy`, `tensorflow`, `argparse`, `shutil`.

---

### 2. **Install Required Libraries**
Run the following command to install the necessary libraries:

```bash
pip install numpy tensorflow
```

---

### 3. **Download the Script**
- Save the script provided earlier as `file_organizer.py` on your computer.

---

### 4. **Run the Script**
Open a terminal or command prompt and navigate to the directory where the script is saved. Run the script using the following command:

```bash
python file_organizer.py --directory /path/to/your/directory
```

- Replace `/path/to/your/directory` with the path to the directory you want to organize.
- If you don't specify a directory, the script will use the current directory (`.`).

---

### 5. **How It Works**
- The agent will scan the directory for files.
- It will move files into folders based on their types (e.g., `.pdf` files to `documents`, `.jpg` files to `images`).
- It will log every file move and folder creation in the terminal.
- The agent will learn from its actions and improve over time.

---

### 6. **Example**
To organize files in the `Downloads` folder, run:

```bash
python file_organizer.py --directory ~/Downloads
```

---

### 7. **Stopping the Script**
- The script will stop automatically after organizing all files.
- To stop it manually, press `Ctrl + C` in the terminal.

---

### 8. **Troubleshooting**
- **Error: No such file or directory**: Ensure the directory path is correct and accessible.
- **Error: TensorFlow not found**: Make sure TensorFlow is installed correctly (`pip install tensorflow`).

---

### 9. **Customization**
- Modify the `file_types` dictionary in the script to add or change file categories.
- Adjust the reward values in the `settings` dictionary to fine-tune the agent's learning.

---

Enjoy an organized directory with the **File Organizer Agent**! 🚀
