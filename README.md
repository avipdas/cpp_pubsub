# cpp_pubsub

A comprehensive ROS 2 example package demonstrating C++ publisher and subscriber nodes, including simulated sensor data publishers for GPS and IMU.

## 🎯 Overview

This package provides practical examples of ROS 2 communication patterns using C++, featuring:

- **Basic Publisher/Subscriber**: Simple text message communication
- **Sensor Simulation**: Realistic GPS and IMU data publishers
- **Multi-Subscriber Pattern**: Single node subscribing to multiple topics
- **Clean Architecture**: Well-structured C++ code following ROS 2 best practices

## 📦 Package Contents

### Core Nodes

| Node | Type | Description |
|------|------|-------------|
| `talker` | Publisher | Publishes "Hello, world!" messages to `/topic` |
| `listener` | Subscriber | Receives and displays messages from `/topic` |
| `gps_publisher` | Publisher | Simulates GPS sensor data on `/gps_topic` |
| `imu_publisher` | Publisher | Simulates IMU sensor data on `/imu_topic` |
| `multi_subscriber` | Subscriber | Listens to both GPS and IMU topics simultaneously |

### Topic Architecture

```
┌─────────────────┐    /topic    ┌─────────────────┐
│     talker      │──────────────▶│    listener     │
└─────────────────┘              └─────────────────┘

┌─────────────────┐  /gps_topic  ┌─────────────────┐
│  gps_publisher  │──────────────▶│                 │
└─────────────────┘              │ multi_subscriber│
                                 │                 │
┌─────────────────┐  /imu_topic  │                 │
│  imu_publisher  │──────────────▶│                 │
└─────────────────┘              └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- ROS 2 Humble (or compatible distribution)
- C++14 compiler
- colcon build tool

### Installation

1. **Clone the package** into your ROS 2 workspace:
   ```bash
   cd ~/ros2_ws/src
   # git clone <your-repository-url>
   ```

2. **Build the package**:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select cpp_pubsub
   ```

3. **Source the workspace** (required for each new terminal):
   ```bash
   source install/local_setup.bash
   ```

## 💻 Usage Examples

### Basic Publisher/Subscriber Demo

**Terminal 1** - Run the publisher:
```bash
ros2 run cpp_pubsub talker
```

**Terminal 2** - Run the subscriber:
```bash
cd ~/ros2_ws
source install/local_setup.bash
ros2 run cpp_pubsub listener
```

### Sensor Simulation Demo

**Terminal 1** - GPS data publisher:
```bash
ros2 run cpp_pubsub gps_publisher
```

**Terminal 2** - IMU data publisher:
```bash
ros2 run cpp_pubsub imu_publisher
```

**Terminal 3** - Multi-subscriber (receives both GPS and IMU):
```bash
ros2 run cpp_pubsub multi_subscriber
```

## 📊 Topic Details

| Topic | Message Type | Publisher | Subscriber(s) | Description |
|-------|-------------|-----------|---------------|-------------|
| `/topic` | `std_msgs::msg::String` | talker | listener | Basic text messages |
| `/gps_topic` | `sensor_msgs::msg::NavSatFix` | gps_publisher | multi_subscriber | Simulated GPS coordinates |
| `/imu_topic` | `sensor_msgs::msg::Imu` | imu_publisher | multi_subscriber | Simulated IMU data |

## 🔧 Development

### File Structure
```
cpp_pubsub/
├── CMakeLists.txt
├── package.xml
├── src/
│   ├── publisher_member_function.cpp    # talker node
│   ├── subscriber_member_function.cpp   # listener node
│   ├── gps_publisher.cpp               # GPS simulation
│   ├── imu_publisher.cpp               # IMU simulation
│   └── multi_subscriber.cpp            # Multi-topic subscriber
└── README.md
```

### Customization

You can easily modify the sensor publishers to:
- Change publication rates
- Simulate different sensor values
- Add new sensor types
- Modify message formats

### Building Individual Nodes

To build only this package:
```bash
colcon build --packages-select cpp_pubsub
```

## 🐛 Troubleshooting

### Common Issues

**Node not found**: Make sure you've sourced the workspace in each terminal:
```bash
source install/local_setup.bash
```

**Build errors**: Ensure all dependencies are installed:
```bash
rosdep install --from-paths src --ignore-src -r -y
```

**No messages received**: Check if publisher and subscriber are running and topics match:
```bash
ros2 topic list
ros2 topic echo /topic
```

## 📝 Additional Commands

### Monitoring Topics
```bash
# List all active topics
ros2 topic list

# Monitor a specific topic
ros2 topic echo /gps_topic

# Check topic information
ros2 topic info /imu_topic
```

### Node Information
```bash
# List running nodes
ros2 node list

# Get node information
ros2 node info /talker
```

## 🤝 Contributing

Feel free to extend this package with:
- Additional sensor types
- More complex message patterns
- Service and action examples
- Parameter configuration

## 📄 License

This package is provided as an educational example for ROS 2 development.

---

> **Note**: Remember to source your workspace (`source install/local_setup.bash`) in each new terminal before running ROS 2 commands.
