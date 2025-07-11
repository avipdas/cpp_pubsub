cpp_pubsub
A simple ROS 2 example package demonstrating C++ publisher and subscriber nodes, including simulated sensor publishers.

🗂️ Overview
This package shows how to:

Create a publisher node that publishes "Hello, world!" messages (talker)

Create a subscriber node that prints those messages (listener)

Create a GPS publisher node simulating fake GPS data

Create an IMU publisher node simulating fake IMU data

Create a multi-subscriber node that subscribes to both GPS and IMU topics

⚡ Setup and Build
Clone or copy this package into your ROS 2 workspace
bash
Copy
Edit
cd ~/ros2_ws/src
# git clone git@github.com:avipdas/cpp_pubsub.git
Go to workspace root and build
bash
Copy
Edit
cd ~/ros2_ws
colcon build
Source the workspace
bash
Copy
Edit
source install/local_setup.bash
⚠️ You need to source this in each new terminal where you want to run nodes.

🚀 Running the nodes
Original talker and listener
Run the talker
bash
Copy
Edit
ros2 run cpp_pubsub talker
Run the listener in a new terminal
bash
Copy
Edit
cd ~/ros2_ws
source install/local_setup.bash
ros2 run cpp_pubsub listener
Simulated GPS and IMU (Phase 2)
Run GPS publisher
bash
Copy
Edit
ros2 run cpp_pubsub gps_publisher
Run IMU publisher
bash
Copy
Edit
ros2 run cpp_pubsub imu_publisher
Run multi-subscriber (listens to GPS and IMU)
bash
Copy
Edit
ros2 run cpp_pubsub multi_subscriber
💬 Topics
Node	Publishes to	Subscribes to
talker	/topic	—
listener	—	/topic
gps_publisher	/gps_topic	—
imu_publisher	/imu_topic	—
multi_subscriber	—	/gps_topic, /imu_topic

✅ Requirements
ROS 2 Humble (or compatible)

C++14

💡 Extra Notes
Always run source install/local_setup.bash after building or in new terminals before running ROS 2 nodes.

You can modify the .cpp files to simulate different data (e.g., new sensor topics).
