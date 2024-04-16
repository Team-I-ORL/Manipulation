#include <memory>
#include "orbiter_bt/srv/move_arm.hpp"
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

#define PLANNINGGROUP "arm"

class MoveArmServer : public rclcpp::Node
{
    public:
    rclcpp::Service<orbiter_bt::srv::MoveArm>::SharedPtr service;
    moveit::planning_interface::MoveGroupInterface move_group; 

    MoveArmServer()
        : Node("move_arm_server"),
        move_group(rclcpp::Node::SharedPtr(this),PLANNINGGROUP)
    {   
        RCLCPP_INFO(get_logger(), "Move Arm Server Created");
        service = this->create_service<orbiter_bt::srv::MoveArm>("move_arm", std::bind(&MoveArmServer::moveArm_callback, this, std::placeholders::_1, std::placeholders::_2));
        RCLCPP_INFO(get_logger(), "Planning frame: %s", move_group.getPlanningFrame().c_str());
    }
    void moveArm_callback(const std::shared_ptr<orbiter_bt::srv::MoveArm::Request> request,
                          std::shared_ptr<orbiter_bt::srv::MoveArm::Response> response)
    {   
        // Adding orientation constraints
        // moveit_msgs::msg::OrientationConstraint ocm;
        // ocm.link_name = "grippper_link";
        // ocm.header.frame_id = "map";
        // ocm.orientation.x = 0.000;
        // ocm.orientation.y = 0.707;
        // ocm.orientation.z = 0.000;
        // ocm.orientation.w = 0.707;
        // ocm.absolute_x_axis_tolerance = 0.1;
        // ocm.absolute_y_axis_tolerance = 0.1;
        // ocm.absolute_z_axis_tolerance = 0.1;
        // ocm.weight = 1.0;

        // moveit_msgs::msg::Constraints test_constraints;
        // test_constraints.orientation_constraints.push_back(ocm);
        // move_group.setPathConstraints(test_constraints);


        if (request->target_pose.header.frame_id == "home"){
            RCLCPP_INFO(get_logger(), "Setting target to home");
            
            move_group.setNamedTarget("home");
        }
        else if (request->target_pose.header.frame_id == "inter_pose_1"){
            RCLCPP_INFO(get_logger(), "Setting target to inter_pose_1");
            move_group.setNamedTarget("inter_pose_1");
        }
        else if (request->target_pose.header.frame_id == "inter_pose_2"){
            RCLCPP_INFO(get_logger(), "Setting target to inter_pose_2");
            move_group.setNamedTarget("inter_pose_2");
        }
        else{
            move_group.setPoseTarget(request->target_pose);
        }

        ///////////////////////////////////////////////////////////////////////////////
        // std::vector<geometry_msgs::msg::Pose> waypoints;

        // // Cartesian Path
        // if (request->target_pose.header.frame_id == "home"){
        //     // If the target pose is "home", set the named target to "home"
        //     move_group.setNamedTarget("home");
        //     waypoints.push_back(move_group.getCurrentPose().pose);
        // }
        // else{
        //     waypoints.push_back(request->target_pose.pose);
        // }
        /////////////////////////////////////////////////////////////////////////////
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        bool success = (move_group.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
        if (success){
            auto result = move_group.execute(my_plan);
            if (result == moveit::core::MoveItErrorCode::SUCCESS){
                response->success = true;
                RCLCPP_INFO(get_logger(), "Move Arm Succeeded");
                return;
            }
            else {
                response->success = false;
                RCLCPP_ERROR(get_logger(), "Move Arm Execution Failed");
                return;
            }
        }
        else {
            response->success = false;
            RCLCPP_ERROR(get_logger(), "Move Arm Planning Failed");
            return;
        }
        }
        ///////////////////////////////////////////////////////////////////////////////
        // moveit_msgs::msg::RobotTrajectory trajectory;
        // const double jump_threshold = 0.0;
        // const double eef_step = 0.01;

        // double fraction = move_group.computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);

        // RCLCPP_INFO(get_logger(), "Visualizing plan (Cartesian path) (%.2f%% acheived)", fraction * 100.0);

        // if (fraction == 1.0) {
        //     moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        //     my_plan.trajectory = trajectory;
        //     auto result = move_group.execute(my_plan);
        //     if (result == moveit::core::MoveItErrorCode::SUCCESS){
        //         response->success = true;
        //         RCLCPP_INFO(get_logger(), "Move Arm Succeeded");
        //         return;
        //     }
        //     else {
        //         response->success = false;
        //         RCLCPP_ERROR(get_logger(), "Move Arm Execution Failed");
        //         return;
        //     }
        // }
        // else {
        //     RCLCPP_ERROR(get_logger(), "Could not compute full Cartesian path");
        // }
    
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto moveArmServer = std::make_shared<MoveArmServer>();
    rclcpp::spin(moveArmServer);
    rclcpp::shutdown();
    return 0;
}