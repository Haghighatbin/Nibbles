# Besty's Saga 🐍

<img align="right" width="200" height="200" src="https://user-images.githubusercontent.com/10771949/155462505-8449f480-0c4f-41bb-96c9-e2b47ca665e9.png">

**Besty's Saga** is a twist on the classic *Snake* game—featuring a cheeky serpent, voice narration, soundtrack, progressive levels, and now an artificial intelligence agent trained to master the game.

This repository includes both the manually controlled version of the game (`betsy.py`) and a fully autonomous DQN-based agent that learns to play and navigate increasingly complex maps through reinforcement learning.

---

## 🧠 AI Mechanism

The artificial intelligence agent is built using **Deep Q-Learning (DQN)** with modern enhancements including:

- **Double DQN**: Prevents Q-value overestimation by decoupling action selection and evaluation.
- **Dueling Architecture**: Separates state-value and advantage estimation for more robust learning.
- **Prioritised Experience Replay**: Samples impactful transitions more frequently.
- **Curriculum Learning**: Trains progressively across 10 levels of increasing obstacle complexity.

The environment is fully custom-built using `pygame`, supporting both rendering and headless modes for training and inference.

---

## 🎮 Game Rules

- Betsy hits a wall? She loses.  
- Betsy collides with herself? She loses.  
- Betsy touches a fence? Electrocution. Game over.  
- Betsy eats 30 apples? A portal opens to the next level.  

Oh, and Betsy *talks*. A lot. You can mute her voice and the soundtrack by clicking the icons at the top-right corner.

Enjoy the game—with or without her sass.

---

## 📁 Project Structure

├── Images/ # Game assets <br/>
├── Sounds/ # Game soundtrack and Betsy's voice <br/>
├── trained_model/ # Saved DQN models after training <br/>
│ <br/>
├── betsy.py # Manual gameplay version (with narration) <br/>
├── nibbles_env.py # Reinforcement learning environment <br/>
├── nibbles_train.py # DQN training loop with curriculum learning <br/>
├── nibbles_play.py # Run a trained agent through the 10 levels <br/>
├── levels_ai.py # Procedural level generation for obstacles <br/>
├── requirements.txt # Python dependencies <br/>
├── LICENSE # MIT License <br/>
└── README.md # This file<br/>

---

## 🛠️ Installation

> ⚠️ Python 3.8 or later is recommended.

```bash
git clone https://github.com/yourusername/bestys-saga.git
cd bestys-saga
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📦 Requirements
- pygame==2.1.2
- pygame-gui==0.6.4
- numpy
- torch
- rich

## 🧪 Training the Agent
Train the AI agent using curriculum learning over 10 levels:
```bash
python nibbles_train.py
```
This script will:

- Begin with small, empty maps.
- Gradually introduce larger grids and more obstacles.
- Save the best model after each level in trained_model/.
- Training can take a few hours depending on system specs.

## 🤖 Running the Trained AI
Once trained, run the agent to play all 10 levels:
```bash
python nibbles_play.py
```
The AI will:
Start at level 0 and progress automatically upon reaching 10 apples.
Handle dynamic obstacles and map sizes.
Use a trained model from trained_model/.
You can modify the model path at the bottom of nibbles_play.py.

## 🎥 Demo
Watch Betsy play with AI:<br/>
<video src="https://github.com/user-attachments/assets/206f8d41-7076-4e73-a778-17a2fd2bd156" width="320" height="240" controls></video>


## 🎵 Soundtrack
The game includes a dynamic background playlist and sound FX:

- Do Your Chain Hang Low – Jibbs
- Fight! – Daniel Asadi
- Immortals – Daniel Asadi
- Her and the Sea – The Clann
- Rue des enfants – Luca Longobardi
- If You Crump Stand Up – EdiT
- Toggle sound via in-game icons.

## 📜 Licence
This project is licensed under the MIT Licence.
Feel free to fork, modify, or contribute.

## 🙋‍♂️ Contributing
Pull requests and issues are welcome!
If you enjoy the project, give it a ⭐ 

## 📫 Contact
Created with love (and sarcasm) by Amin Haghighatbin <br />
📧 [aminhb@tutanota.com]<br /><br />
🐍 Long live Betsy!<br />