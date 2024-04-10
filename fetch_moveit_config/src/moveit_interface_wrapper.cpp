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
        move_group.setPoseTarget(request->target_pose);
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
    };

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto moveArmServer = std::make_shared<MoveArmServer>();
    rclcpp::spin(moveArmServer);
    rclcpp::shutdown();
    return 0;
}