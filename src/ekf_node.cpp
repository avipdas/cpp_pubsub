#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <Eigen/Dense>
#include <cuda_runtime.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

extern "C" void launch_kalman_update(float* x, float* P, const float* gps, float* K_out);


class EKFNode : public rclcpp::Node
{
public:
  EKFNode() : Node("ekf_node")
  {
    RCLCPP_INFO(this->get_logger(), "EKF Node initialized.");
    gps_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
      "/gps_topic", 10, std::bind(&EKFNode::gps_callback, this, std::placeholders::_1));

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "/imu_topic", 10, std::bind(&EKFNode::imu_callback, this, std::placeholders::_1));

    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("filtered_pose", 10);

    x_ = VectorXd::Zero(4);
    P_ = MatrixXd::Identity(4, 4);

    last_time_ = this->now();
  }

private:
  VectorXd x_;
  MatrixXd P_;
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

    P_ = P_ + MatrixXd::Identity(4, 4) * 0.1;
  }

  void update(double gps_x, double gps_y)
  {
    float x_host[4] = {static_cast<float>(x_(0)), static_cast<float>(x_(1)), static_cast<float>(x_(2)), static_cast<float>(x_(3))};
    float P_host[16];
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        P_host[i * 4 + j] = static_cast<float>(P_(i, j));
    float gps_host[2] = {static_cast<float>(gps_x), static_cast<float>(gps_y)};
    //float K_host[8] = {0};

    float *x_dev, *P_dev, *gps_dev, *K_dev;
    cudaMalloc(&x_dev, 4 * sizeof(float));
    cudaMalloc(&P_dev, 16 * sizeof(float));
    cudaMalloc(&gps_dev, 2 * sizeof(float));
    cudaMalloc(&K_dev, 8 * sizeof(float));

    cudaMemcpy(x_dev, x_host, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(P_dev, P_host, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gps_dev, gps_host, 2 * sizeof(float), cudaMemcpyHostToDevice);

    launch_kalman_update(x_dev, P_dev, gps_dev, K_dev);
    cudaDeviceSynchronize();

    cudaMemcpy(x_host, x_dev, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    x_(0) = x_host[0];
    x_(1) = x_host[1];
    x_(2) = x_host[2];
    x_(3) = x_host[3];

    cudaFree(x_dev); cudaFree(P_dev); cudaFree(gps_dev); cudaFree(K_dev);

    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = this->now();
    pose_msg.pose.position.x = x_(0);
    pose_msg.pose.position.y = x_(1);
    pose_msg.pose.position.z = 0.0;
    pose_pub_->publish(pose_msg);
  }

  void gps_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
  {
    RCLCPP_INFO(this->get_logger(), "Received GPS: lat=%f, lon=%f", msg->latitude, msg->longitude);
    update(msg->latitude, msg->longitude);
  }

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    rclcpp::Time current_time = this->now();
    double dt = (current_time - last_time_).seconds();
    last_time_ = current_time;

    double ax = msg->linear_acceleration.x;
    double ay = msg->linear_acceleration.y;

    RCLCPP_INFO(this->get_logger(), "Received IMU: ax=%f, ay=%f, dt=%f", ax, ay, dt);
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
