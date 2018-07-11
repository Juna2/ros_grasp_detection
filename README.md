# Detecting grasping positions with deep neural networks using RGB images


- Use grasp_detection for training model. This is for test.
<br />
*grasp_detection* Link : https://github.com/Juna2/grasp_detection

- Most of the files in this repo is from his work except test.py

- test.py loads pretrained model in 

    /home/<your_path>/catkin_ws/src/ros_grasp_detection/src/m4

- *you may change the path in test.py*

- I recommand you to train model with Juna2/grasp_detection(which is also modified version of robot-grasp-detection) first and copy the model to the path.





## How to use

1. $ catkin_make

2.  
    $ roslaunch ros_grasp_detection

3. It subscribes "/croppedRoI" message which is 224x224 object image and publishes "/objects" which contains center x, y and degree. So you have to use another package to publish 224x224 object image to this package.

4. The grasp detection result will be saved in      

    /home/<your_path>/catkin_ws/src/ros_grasp_detection/src/image/bbox.jpg


- **One Thing you should know is that this test.py uses the model trained by modified robot-grasp-detection which is Juna2/grasp_detection.**



## sample

<img src=./src/image/example/bbox0.jpg width="30%"> <img src=./src/image/example/bbox1.jpg width="30%">
<br />
<img src=./src/image/example/bbox2.jpg width="30%"> <img src=./src/image/example/bbox3.jpg width="30%">

