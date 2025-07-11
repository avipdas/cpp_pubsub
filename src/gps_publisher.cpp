#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"

using namespace std::chrono_literals;

class FakeGPSPublisher : public rclcpp::Node
{
public:
  FakeGPSPublisher()
  : Node("fake_gps_publisher")
  {
    publisher_ = this->create_publisher<sensor_msgs::msg::NavSatFix>("gps_topic", 10);
    timer_ = this->create_wall_timer(100ms, std::bind(&FakeGPSPublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    auto msg = sensor_msgs::msg::NavSatFix();
    msg.latitude = 37.7749 + 0.001 * count_;
    msg.longitude = -122.4194 + 0.001 * count_;
    msg.altitude = 10.0;
    RCLCPP_INFO(this->get_logger(), "Publishing GPS: lat=%f, lon=%f", msg.latitude, msg.longitude);
    publisher_->publish(msg);
    count_++;
  }

  rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  int count_ = 0;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FakeGPSPublisher>());
  rclcpp::shutdown();
  return 0;
}
