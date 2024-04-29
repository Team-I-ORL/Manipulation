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

        // Set planning time and number of retries
        // double named_planning_time = 30.0;  // Set your desired planning time here
        double planning_time = 120.0;
        int num_retries = 5;  // Set your desired number of retries here
        move_group.setPlanningTime(planning_time);
        move_group.setNumPlanningAttempts(num_retries);
        move_group.setWorkspace(-2, -2, 0.2, 2, 2, 2);
        // move_group.setStartStateTolerance(0.1);
        // move_group.TrajectoryExecutionManager::allowedStartTolerance(0.1);

        // Set goal joint tolerance
        double goal_joint_tolerance = 0.1;  // Set your desired goal joint tolerance here
        move_group.setGoalJointTolerance(goal_joint_tolerance);

        double velocity_scaling_factor = 1.0;  // Set your desired velocity scaling factor here
        move_group.setMaxVelocityScalingFactor(velocity_scaling_factor);

        ////// 
        // this->set_parameter(rclcpp::Parameter("trajectory_execution/allowed_execution_duration_scaling", 2.0));
    }
    void moveArm_callback(const std::shared_ptr<orbiter_bt::srv::MoveArm::Request> request,
                          std::shared_ptr<orbiter_bt::srv::MoveArm::Response> response)
    {   
        move_group.setStartStateToCurrentState();
        int move_id = 1;
        // double named_planning_time = 120.0;  // Set your desired planning time here
        // double planning_time = 120.0;
        RCLCPP_INFO(get_logger(), "Calling back moveArm_callback!!");
        
        // if (request->target_pose.header.frame_id == "home"){
        //     RCLCPP_INFO(get_logger(), "Setting target to home");
            
        //     move_group.clearPathConstraints();
        //     move_group.setNamedTarget("home");
        // }
        // else if (request->target_pose.header.frame_id == "inter_pose_1"){
        //     RCLCPP_INFO(get_logger(), "Setting target to inter_pose_1");
        //     move_group.clearPathConstraints();
        //     move_group.setNamedTarget("inter_pose_1");
        // }
        // else if (request->target_pose.header.frame_id == "inter_pose_2"){
        //     RCLCPP_INFO(get_logger(), "Setting target to inter_pose_2");
        //     move_group.clearPathConstraints();
        //     move_group.setNamedTarget("inter_pose_2");
        // }
        // else{
        //     // Adding orientation constraints
        //     moveit_msgs::msg::OrientationConstraint ocm;
        //     ocm.link_name = "fetch_tool_link";
        //     ocm.header.frame_id = "base_link";
        //     ocm.orientation.x = 0.000;
        //     ocm.orientation.y = 0.000;//0.707;
        //     ocm.orientation.z = 0.000;
        //     ocm.orientation.w = 1.000;//0.707;
        //     ocm.absolute_x_axis_tolerance = 0.5;
        //     ocm.absolute_y_axis_tolerance = 0.5;
        //     ocm.absolute_z_axis_tolerance = 0.5;
        //     ocm.weight = 0.5;

        //     moveit_msgs::msg::Constraints test_constraints;
        //     test_constraints.orientation_constraints.push_back(ocm);
        //     move_group.setPathConstraints(test_constraints);

        //     move_group.setPoseTarget(request->target_pose, "fetch_tool_link");
        // }

        ///////////////////////////////////////////////////////////////////////////////
        std::vector<geometry_msgs::msg::Pose> waypoints;

        // Cartesian Path
        if (request->target_pose.header.frame_id == "home"){
            // If the target pose is "home", set the named target to "home"
            RCLCPP_INFO(get_logger(), "Setting target to home");
            // move_group.setPlanningTime(named_planning_time);

            // moveit_msgs::msg::OrientationConstraint ocm;
            // ocm.link_name = "fetch_tool_link";
            // ocm.header.frame_id = "base_link";
            // ocm.orientation.x = 0.000;
            // ocm.orientation.y = 0.000;//0.707;
            // ocm.orientation.z = 0.000;
            // ocm.orientation.w = 1.000;//0.707;
            // ocm.absolute_x_axis_tolerance = 0.5;
            // ocm.absolute_y_axis_tolerance = 0.5;
            // ocm.absolute_z_axis_tolerance = 0.5;
            // ocm.weight = 0.5;

            // moveit_msgs::msg::Constraints test_constraints;
            // test_constraints.orientation_constraints.push_back(ocm);
            // move_group.setPathConstraints(test_constraints);

            move_group.setNamedTarget("home");
            move_id = 0;
            waypoints.push_back(move_group.getCurrentPose().pose);
        }
        else if (request->target_pose.header.frame_id == "inter_pose_1"){
            RCLCPP_INFO(get_logger(), "Setting target to inter_pose_1");
            move_group.clearPathConstraints();
            // move_group.setPlanningTime(named_planning_time);


            // moveit_msgs::msg::OrientationConstraint ocm;
            // ocm.link_name = "fetch_tool_link";
            // ocm.header.frame_id = "base_link";
            // ocm.orientation.x = 0.000;
            // ocm.orientation.y = 0.000;//0.707;
            // ocm.orientation.z = 0.000;
            // ocm.orientation.w = 1.000;//0.707;
            // ocm.absolute_x_axis_tolerance = 0.5;
            // ocm.absolute_y_axis_tolerance = 0.5;
            // ocm.absolute_z_axis_tolerance = 0.5;
            // ocm.weight = 0.5;

            // moveit_msgs::msg::Constraints test_constraints;
            // test_constraints.orientation_constraints.push_back(ocm);
            // move_group.setPathConstraints(test_constraints);
            

            move_group.setNamedTarget("inter_pose_1");
            move_id = 0;
            waypoints.push_back(move_group.getCurrentPose().pose);
        }
        else if (request->target_pose.header.frame_id == "inter_pose_2"){
            RCLCPP_INFO(get_logger(), "Setting target to inter_pose_2");
            move_group.clearPathConstraints();
            // move_group.setPlanningTime(named_planning_time);


            // moveit_msgs::msg::OrientationConstraint ocm;
            // ocm.link_name = "fetch_tool_link";
            // ocm.header.frame_id = "base_link";
            // ocm.orientation.x = 0.000;
            // ocm.orientation.y = 0.000;//0.707;
            // ocm.orientation.z = 0.000;
            // ocm.orientation.w = 1.000;//0.707;
            // ocm.absolute_x_axis_tolerance = 0.5;
            // ocm.absolute_y_axis_tolerance = 0.5;
            // ocm.absolute_z_axis_tolerance = 0.5;
            // ocm.weight = 0.5;

            // moveit_msgs::msg::Constraints test_constraints;
            // test_constraints.orientation_constraints.push_back(ocm);
            // move_group.setPathConstraints(test_constraints);
            
            move_group.setNamedTarget("inter_pose_2");
            move_id = 0;
            waypoints.push_back(move_group.getCurrentPose().pose);
        }
        else{
            RCLCPP_INFO(get_logger(), "Setting target to custom pose");
            // move_group.setPlanningTime(planning_time);
            move_id = 1;  
            waypoints.push_back(request->target_pose.pose);
        }


        /////////////////////////////////////////////////////////////////////////////
        while(1){
            if (move_id == 0){

            //Delay 
            rclcpp::sleep_for(std::chrono::seconds(5));

            moveit::planning_interface::MoveGroupInterface::Plan my_plan;
            bool success = (move_group.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
            if (success){
                auto result = move_group.execute(my_plan);
                if (result == moveit::core::MoveItErrorCode::SUCCESS){
                    response->success = true;
                    RCLCPP_INFO(get_logger(), "Move Arm Succeeded when moving to named target");
                    return;
                }
                else {
                    response->success = false;
                    RCLCPP_ERROR(get_logger(), "Move Arm Execution Failed when moving to named target");
                    return;
                }
            }
            else {
                response->success = false;
                RCLCPP_ERROR(get_logger(), "Move Arm Planning Failed when moving to named target");
                // return;
                move_id = 1;
            }
        
        }
        else{
            RCLCPP_INFO(get_logger(), "Computing Cartesian Path");
            
            ///////////////////////////////////////////////////////////////////////////////
            moveit_msgs::msg::RobotTrajectory trajectory;
            const double jump_threshold = 0.0;
            const double eef_step = 0.05;

            double fraction = move_group.computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);

            RCLCPP_INFO(get_logger(), "Visualizing plan (Cartesian path) (%.2f%% acheived)", fraction * 100.0);
            
            

            if (fraction == 1.0) {
                moveit::planning_interface::MoveGroupInterface::Plan my_plan;
                my_plan.trajectory = trajectory;
                auto result = move_group.execute(my_plan);
                rclcpp::sleep_for(std::chrono::seconds(5));

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
                RCLCPP_ERROR(get_logger(), "Could not compute full Cartesian path");
                return;
            }
        }
        move_group.clearPathConstraints();
    }
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