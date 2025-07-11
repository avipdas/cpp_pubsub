#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include <memory>

using namespace std::chrono_literals;

class FakeIMUPublisher : public rclcpp::Node
{
public:
  FakeIMUPublisher()
  : Node("fake_imu_publisher")
  {
    publisher_ = this->create_publisher<sensor_msgs::msg::Imu>("imu_topic", 10);
    timer_ = this->create_wall_timer(2ms, std::bind(&FakeIMUPublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    auto msg = sensor_msgs::msg::Imu();
    msg.angular_velocity.x = 0.1;
    msg.angular_velocity.y = 0.2;
    msg.angular_velocity.z = 0.3;
    msg.linear_acceleration.x = 0.0;
    msg.linear_acceleration.y = 9.8;
    msg.linear_acceleration.z = 0.0;
    RCLCPP_INFO(this->get_logger(), "Publishing IMU: ang_vel_z=%f", msg.angular_velocity.z);
    publisher_->publish(msg);
  }

  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FakeIMUPublisher>());
  rclcpp::shutdown();
  return 0;
}
