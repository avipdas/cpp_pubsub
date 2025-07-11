#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include <memory>

class FusionNode : public rclcpp::Node
{
public:
  FusionNode()
  : Node("fusion_node"), gps_received_(false), imu_received_(false)
  {
    gps_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
      "gps_topic", 10,
      [this](const sensor_msgs::msg::NavSatFix::SharedPtr msg) {
        last_gps_ = *msg;
        gps_received_ = true;
        try_fuse();
      });

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "imu_topic", 10,
      [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
        last_imu_ = *msg;
        imu_received_ = true;
        try_fuse();
      });
  }

private:
  void try_fuse()
  {
    if (gps_received_ && imu_received_) {
      double fused_lat = last_gps_.latitude; // Simple example: just using GPS
      double fused_lon = last_gps_.longitude;

      RCLCPP_INFO(this->get_logger(),
        "Fused Data: lat=%.6f, lon=%.6f, IMU ang_vel_z=%.2f",
        fused_lat, fused_lon, last_imu_.angular_velocity.z);

      gps_received_ = false;
      imu_received_ = false;
    }
  }

  sensor_msgs::msg::NavSatFix last_gps_;
  sensor_msgs::msg::Imu last_imu_;
  bool gps_received_;
  bool imu_received_;

  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FusionNode>());
  rclcpp::shutdown();
  return 0;
}
