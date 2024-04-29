#include <memory>
#include "orbiter_bt/srv/move_arm.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

#define PLANNINGGROUP "arm"

class MoveArmServer : public rclcpp::Node
{
    public:
    rclcpp::Service<orbiter_bt::srv::MoveArm>::SharedPtr service;
    moveit::planning_interface::MoveGroupInterface move_group; 
    //Define a publisher to publish on /isaac_joint_command topic

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_command_publisher;

    MoveArmServer()
        : Node("move_arm_server"),
        move_group(rclcpp::Node::SharedPtr(this),PLANNINGGROUP)
    {   
        RCLCPP_INFO(get_logger(), "Move Arm Server Created");
        service = this->create_service<orbiter_bt::srv::MoveArm>("move_arm", std::bind(&MoveArmServer::moveArm_callback, this, std::placeholders::_1, std::placeholders::_2));
        RCLCPP_INFO(get_logger(), "Planning frame: %s", move_group.getPlanningFrame().c_str());

        // Set planning time and number of retries
        double planning_time = 120.0;
        int num_retries = 5;  // Set your desired number of retries here
        move_group.setPlanningTime(planning_time);
        move_group.setNumPlanningAttempts(num_retries);
        move_group.setWorkspace(-2, -2, 0.2, 2, 2, 2);

        // Set goal joint tolerance
        double goal_joint_tolerance = 0.1; // Set your desired goal joint tolerance here
        move_group.setGoalJointTolerance(goal_joint_tolerance);

        double velocity_scaling_factor = 1.0; // Set your desired velocity scaling factor here
        move_group.setMaxVelocityScalingFactor(velocity_scaling_factor);

        // Define a publisher to publish on /isaac_joint_command topic
        joint_command_publisher = this->create_publisher<sensor_msgs::msg::JointState>("/isaac_joint_commands", 10);

    }
    void moveArm_callback(const std::shared_ptr<orbiter_bt::srv::MoveArm::Request> request,
                          std::shared_ptr<orbiter_bt::srv::MoveArm::Response> response)
    {   
        RCLCPP_INFO(get_logger(), "Calling back moveArm_callback!!");
        
        if (request->target_pose.header.frame_id == "home"){
            RCLCPP_INFO(get_logger(), "Setting target to home");
            // Publish on /isaac_joint_command topic with joint name as home
            sensor_msgs::msg::JointState joint_command;
            joint_command.name.push_back("home");
            // Publish the joint command
            joint_command_publisher->publish(joint_command);
            // Sleep on the main thread for 5 seconds
            rclcpp::sleep_for(std::chrono::seconds(2));
            response->success = true;
            return;
        }
        else if (request->target_pose.header.frame_id == "inter_pose_1"){
            RCLCPP_INFO(get_logger(), "Setting target to inter_pose_1");
            // Publish on /isaac_joint_command topic with joint name as home
            sensor_msgs::msg::JointState joint_command;
            joint_command.name.push_back("inter_pose_1");
            // Publish the joint command
            joint_command_publisher->publish(joint_command);
            // Sleep on the main thread for 5 seconds
            rclcpp::sleep_for(std::chrono::seconds(2));
            response->success = true;
            return;
        }
        else if (request->target_pose.header.frame_id == "inter_pose_2"){
            RCLCPP_INFO(get_logger(), "Setting target to inter_pose_2");
            // Publish on /isaac_joint_command topic with joint name as home
            sensor_msgs::msg::JointState joint_command;
            joint_command.name.push_back("inter_pose_2");
            // Publish the joint command
            joint_command_publisher->publish(joint_command);
            // Sleep on the main thread for 5 seconds
            rclcpp::sleep_for(std::chrono::seconds(2));
            response->success = true;
            return;
        }
        else{
            std::vector<geometry_msgs::msg::Pose> waypoints;
            RCLCPP_INFO(get_logger(), "Going to Aurco Marker Pose");
            waypoints.push_back(request->target_pose.pose);

            ///////////////////////////////////////////////////////////////////////////////
            moveit_msgs::msg::RobotTrajectory trajectory;
            const double jump_threshold = 0.0;
            const double eef_step = 0.05;

            double fraction = move_group.computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);

            RCLCPP_INFO(get_logger(), "Visualizing plan (Cartesian path) (%.2f%% acheived)", fraction * 100.0);
            
            if (fraction == 1.0) {
                moveit::planning_interface::MoveGroupInterface::Plan my_plan;
                my_plan.trajectory = trajectory;

                // Publish the joint command
                sensor_msgs::msg::JointState joint_command;

                // Extract the joint values from the trajectory
                for (size_t i = 0; i < trajectory.joint_trajectory.points.size(); i++){
                    joint_command.name = trajectory.joint_trajectory.joint_names;
                    joint_command.position = trajectory.joint_trajectory.points[i].positions;
                    // Print the joint_command message
                    // for (size_t j = 0; j < joint_command.name.size(); j++){
                    //     RCLCPP_INFO(get_logger(), "Joint Name: %s, Joint Position: %f", joint_command.name[j].c_str(), joint_command.position[j]);
                    // }
                    // RCLCPP_INFO(get_logger(), "");
                    
                    joint_command_publisher->publish(joint_command);
                    rclcpp::sleep_for(std::chrono::milliseconds(100));
                }
                response->success = true;
                RCLCPP_INFO(get_logger(), "Move Arm Succeeded");
                return;


                // auto result = move_group.execute(my_plan);
                // rclcpp::sleep_for(std::chrono::seconds(5));

                // if (result == moveit::core::MoveItErrorCode::SUCCESS){
                //     response->success = true;
                //     RCLCPP_INFO(get_logger(), "Move Arm Succeeded");
                //     return;
                // }
                // else {
                //     response->success = false;
                //     RCLCPP_ERROR(get_logger(), "Move Arm Execution Failed");
                //     return;
                // }
            }
            else {
                response->success = false;
                RCLCPP_ERROR(get_logger(), "Could not compute full Cartesian path");
                return;
            }
        }
        move_group.clearPathConstraints();
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto moveArmServer = std::make_shared<MoveArmServer>();
    rclcpp::spin(moveArmServer);
    rclcpp::shutdown();
    return 0;
}