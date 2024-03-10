## Building Moveit2 from source

1. Source ROS

   ```
   source /opt/ros/humble/setup.bash
   ```

2. Install rosdep

   ```
   sudo apt install python3-rosdep
   ```

3. Once you have ROS 2 installed, make sure you have the most up to date packages:

   ```
   sudo rosdep init
   rosdep update
   sudo apt update
   sudo apt dist-upgrade
   ```

4. Install Colcon:

   ```
   sudo apt install python3-colcon-common-extensions
   sudo apt install python3-colcon-mixin
   colcon mixin add default https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml
   colcon mixin update default
   ```

5. Install vcstool:

   ```
   sudo apt install python3-vcstool
   ```

6. Create moveit workspace directory

   ```
   mkdir -p ~/ws_moveit/src
   cd ~/ws_moveit/src
   git clone https://github.com/ros-planning/moveit2_tutorials
   ```

7. Download rest of the source code:

   ```
   vcs import < moveit2_tutorials/moveit2_tutorials.repos
   ```

8. Install all of moveit dependencies

   ```
   sudo apt update && rosdep install -r --from-paths . --ignore-src --rosdistro $ROS_DISTRO -y
   ```

9. Build moveit (**IMP**)

   ```
   colcon build --event-handlers desktop_notification- status- --cmake-args -DCMAKE_BUILD_TYPE=Release --executor sequential
   ```

10. Source 

    ```
    source ~/ws_moveit/install/setup.bash
    ```

11. Finally clone this repo into **ws_moveit/src** and run the build command again (step 9)