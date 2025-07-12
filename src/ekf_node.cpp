#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class EKFNode : public rclcpp::Node
{
public:
  EKFNode() : Node("ekf_node")
  {
    gps_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
      "gps_topic", 10,
      std::bind(&EKFNode::gps_callback, this, std::placeholders::_1));

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "imu_topic", 10,
      std::bind(&EKFNode::imu_callback, this, std::placeholders::_1));

    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("filtered_pose", 10);

    // Initialize state vector [x, y, vx, vy]
    x_ = VectorXd::Zero(4);
    P_ = MatrixXd::Identity(4, 4);

    last_time_ = this->now();
  }

private:
  VectorXd x_;   // state vector
  MatrixXd P_;   // covariance

  rclcpp::Time last_time_;

  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;

  void predict(double ax, double ay, double dt)
  {
    x_(0) += x_(2) * dt + 0.5 * ax * dt * dt;
    x_(1) += x_(3) * dt + 0.5 * ay * dt * dt;
    x_(2) += ax * dt;
    x_(3) += ay * dt;

    // Simplified covariance update
    P_ = P_ + MatrixXd::Identity(4, 4) * 0.1;
  }

  void update(double gps_x, double gps_y)
  {
    VectorXd z(2);
    z << gps_x, gps_y;

    VectorXd y = z - x_.head(2);
    MatrixXd H = MatrixXd::Zero(2, 4);
    H(0, 0) = 1;
    H(1, 1) = 1;

    MatrixXd R = MatrixXd::Identity(2, 2) * 0.5;
    MatrixXd S = H * P_ * H.transpose() + R;
    MatrixXd K = P_ * H.transpose() * S.inverse();

    x_ = x_ + K * y;
    P_ = (MatrixXd::Identity(4, 4) - K * H) * P_;

    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = this->now();
    pose_msg.pose.position.x = x_(0);
    pose_msg.pose.position.y = x_(1);
    pose_msg.pose.position.z = 0.0;
    pose_pub_->publish(pose_msg);
  }

  void gps_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
  {
    update(msg->latitude, msg->longitude);
  }

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    rclcpp::Time current_time = this->now();
    double dt = (current_time - last_time_).seconds();
    last_time_ = current_time;

    double ax = msg->linear_acceleration.x;
    double ay = msg->linear_acceleration.y;

    predict(ax, ay, dt);
  }
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EKFNode>());
  rclcpp::shutdown();
  return 0;
}
