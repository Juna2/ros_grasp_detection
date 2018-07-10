#!/usr/bin/env python
import sys
import rospy
import numpy as np
import tensorflow as tf
import grasp_img_proc
from grasp_inf import inference
# from grasp_det import grasp_to_bbox
import cv2
from cv_bridge import CvBridge, CvBridgeError 
from ros_grasp_detection.msg import positionNpose
from sensor_msgs.msg import Image
import math
import sys 


class detector: 
    def __init__(self):
        # self.image_final = np.array(1, 224, 224, 3)
        self.bridge = CvBridge()
        self.count = 0

        self.X = tf.placeholder(tf.float32, [1,224,224,3], name="X")
        self.Y = tf.unstack(inference(self.X), axis=1)

        dg={}
        lg = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4', 'w5', 'b5', 'w_fc1', 'b_fc1', 'w_fc2', 'b_fc2', 'w_output', 'b_output']
        # lg = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4', 'w5', 'b5', 'w_fc1', 'b_fc1', 'w_fc2', 'b_fc2', 'w_fc3', 'b_fc3', 'w_output', 'b_output']
        for i in lg:
            dg[i] = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == i+':0'][0]
        self.saver_g = tf.train.Saver(dg)

        self._session = tf.InteractiveSession()
        
        self.saver_g.restore(self._session, '/home/juna/catkin_ws/src/ros_grasp_detection/src/models/grasp/m4/m4.ckpt')
        # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # self._session.run(init_op)

        self.object_pub = rospy.Publisher("/objects", positionNpose, queue_size=1)
        self.image_sub = rospy.Subscriber("/croppedRoI", Image, self.image_callback, queue_size=1, buff_size=2**24)

    def draw_bbox(self, img, bbox):
        p1 = (int(float(bbox[0][0])), int(float(bbox[0][1])))
        p2 = (int(float(bbox[1][0])), int(float(bbox[1][1])))
        p3 = (int(float(bbox[2][0])), int(float(bbox[2][1])))
        p4 = (int(float(bbox[3][0])), int(float(bbox[3][1])))
        cv2.line(img, p1, p2, (255, 0, 0))
        cv2.line(img, p1, p3, (0, 0, 255))
        cv2.line(img, p4, p3, (0, 0, 255))
        cv2.line(img, p4, p2, (0, 0, 255))

        return img

    def grasp_to_bbox(self, x, y, x_vec, y_vec, h):
        hypo = math.sqrt(x_vec**2+y_vec**2)
        center = np.array([[1, 0, 0, x],
                           [0, 1, 0, y],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        rotate = np.array([[x_vec/hypo, -y_vec/hypo,   0,   0],
                           [y_vec/hypo,  x_vec/hypo,   0,   0],
                           [         0,           0,   1,   0],
                           [         0,           0,   0,   1]])
        trans_1 = np.array([[1, 0, 0,  h/2],
                            [0, 1, 0,    0],
                            [0, 0, 1,    0],
                            [0, 0, 0,    1]])
        trans_2 = np.array([[1, 0, 0, -h/2],
                            [0, 1, 0,    0],
                            [0, 0, 1,    0],
                            [0, 0, 0,    1]])
        trans_3 = np.array([[1, 0, 0,   0],
                            [0, 1, 0,  20],
                            [0, 0, 1,   0],
                            [0, 0, 0,   1]])
        trans_4 = np.array([[1, 0, 0,   0],
                            [0, 1, 0, -20],
                            [0, 0, 1,   0],
                            [0, 0, 0,   1]])
        point1 = [center.dot(rotate).dot(trans_1).dot(trans_3)[0,3], center.dot(rotate).dot(trans_1).dot(trans_3)[1,3]]
        point2 = [center.dot(rotate).dot(trans_1).dot(trans_4)[0,3], center.dot(rotate).dot(trans_1).dot(trans_4)[1,3]]
        point3 = [center.dot(rotate).dot(trans_2).dot(trans_3)[0,3], center.dot(rotate).dot(trans_2).dot(trans_3)[1,3]]
        point4 = [center.dot(rotate).dot(trans_2).dot(trans_4)[0,3], center.dot(rotate).dot(trans_2).dot(trans_4)[1,3]]

        
    # def grasp_to_bbox(self, x, y, tan, h, w):
    #     theta = math.atan(tan)
    #     edge1 = (x -w/2*math.cos(theta) +h/2*math.sin(theta), y -w/2*math.sin(theta) -h/2*math.cos(theta))
    #     edge2 = (x +w/2*math.cos(theta) +h/2*math.sin(theta), y +w/2*math.sin(theta) -h/2*math.cos(theta))
    #     edge3 = (x +w/2*math.cos(theta) -h/2*math.sin(theta), y +w/2*math.sin(theta) +h/2*math.cos(theta))
    #     edge4 = (x -w/2*math.cos(theta) -h/2*math.sin(theta), y -w/2*math.sin(theta) +h/2*math.cos(theta))      
      
        # edge1 = (x -w/2*math.sin(theta) -h/2*math.cos(theta), y -w/2*math.cos(theta) +h/2*math.sin(theta))
        # edge2 = (x +w/2*math.sin(theta) -h/2*math.cos(theta), y +w/2*math.cos(theta) +h/2*math.sin(theta))
        # edge3 = (x +w/2*math.sin(theta) +h/2*math.cos(theta), y +w/2*math.cos(theta) -h/2*math.sin(theta))
        # edge4 = (x -w/2*math.sin(theta) +h/2*math.cos(theta), y -w/2*math.cos(theta) -h/2*math.sin(theta))


        return [point1, point2, point3, point4]


    def image_callback(self, data):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e, '!')
        img_resize = cv2.resize(cv_image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        
        image_np = np.array([np.asarray(img_resize)])
        # print(image_np)
        image_np = image_np * 1.0 / 255
        image_np = (image_np - 0.5) * 2

        '''There is some images which have 4 as a 3rd vector size like, 
        shape = (224, 224, 4) so you have to slice it so that they become like shape = (224, 224, 3)'''
        # img_slice = tf.slice(img_resize, [0,0,0], [224,224,3]) #### 
        # # print('2 : ', sess.run(img_reshape).shape)
        # img_reshape = tf.reshape(img_slice, shape=[1, 224, 224, 3])
        # cv2.imshow('hahaha', self._session.run(img_reshape)) 
        # cv2.waitKey(0) 
        # print('3 : ', img_reshape.get_shape())
        # x_hat, y_hat, tan_hat, w_hat, h_hat = tf.unstack(inference(img_reshape), axis=1)

        
        predict = self._session.run(self.Y, feed_dict={self.X:image_np})
        print 'x= ', predict[0], '\n y= ', predict[1], '\n x_vec= ', predict[2], '\n y_vec= ', predict[3], '\n h= ', predict[4]
        bbox_hat = self.grasp_to_bbox(predict[0], predict[1], predict[2], predict[3], predict[4]) # (math.atan(predict[2]))*180/3.14
        

        # n = bbox_hat[0][0].eval()
        # print(predict[0])
        img_final = self.draw_bbox(img_resize, bbox_hat)

        cv2.imwrite('/home/juna/catkin_ws/src/ros_grasp_detection/src/image/bbox'+str(self.count)+'.jpg', img_final)
        self.count += 1
        # cv2.waitKey(0)

        self.object_pub.publish(positionNpose(None, predict[0], predict[1], predict[2]))


def main(args):
    rospy.init_node('test')
    # print('4')
    obj=detector()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("ShutDown")
        cv2.destroyAllWindows()

if __name__=='__main__':
    # print('3')
    main(sys.argv)
    # sess.close()


