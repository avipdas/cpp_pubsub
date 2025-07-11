#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "sensor_msgs/msg/imu.hpp"

class MultiSensorSubscriber : public rclcpp::Node
{
public:
  MultiSensorSubscriber()
  : Node("multi_sensor_subscriber")
  {
    gps_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
      "gps_topic", 10, std::bind(&MultiSensorSubscriber::gps_callback, this, std::placeholders::_1));
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "imu_topic", 10, std::bind(&MultiSensorSubscriber::imu_callback, this, std::placeholders::_1));
  }

private:
  void gps_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
  {
    RCLCPP_INFO(this->get_logger(), "Received GPS: lat=%f, lon=%f", msg->latitude, msg->longitude);
  }

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    RCLCPP_INFO(this->get_logger(), "Received IMU: ang_vel_z=%f", msg->angular_velocity.z);
  }

  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MultiSensorSubscriber>());
  rclcpp::shutdown();
  return 0;
}
