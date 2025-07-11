# cpp_pubsub

A simple ROS 2 example package demonstrating a **C++ publisher (talker)** and **subscriber (listener)** node.

---

## 🗂️ Overview

This package shows how to:

- Create a publisher node that publishes "Hello, world!" messages (`talker`)
- Create a subscriber node that listens and prints those messages (`listener`)

---

## ⚡ Setup and Build

### Clone or copy this package into your ROS 2 workspace

```bash
cd ~/ros2_ws/src
# git clone git@github.com:avipdas/cpp_pubsub.git
Go to workspace root and build
bashcd ~/ros2_ws
colcon build
Source the workspace
After building, source your workspace setup file:
bashsource install/local_setup.bash
⚠️ You need to source this in each new terminal where you want to run nodes.

🚀 Running the nodes
Run the publisher (talker)
bashros2 run cpp_pubsub talker
Run the subscriber (listener) in a new terminal
1️⃣ Open a new terminal
2️⃣ Go to workspace root and source:
bashcd ~/ros2_ws
source install/local_setup.bash
3️⃣ Run:
bashros2 run cpp_pubsub listener

💬 Topics

The talker publishes to: /topic
The listener subscribes to: /topic

You can also echo the topic messages in a separate terminal:
bashros2 topic echo /topic

✅ Requirements

ROS 2 Humble (or compatible)
C++14


💡 Extra Notes

Always run source install/local_setup.bash after building or in new terminals before running ROS 2 nodes.
You can modify the .cpp files to change message content or topics to experiment further.
