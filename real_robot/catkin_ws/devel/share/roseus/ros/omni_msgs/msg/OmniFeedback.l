;; Auto-generated. Do not edit!


(when (boundp 'omni_msgs::OmniFeedback)
  (if (not (find-package "OMNI_MSGS"))
    (make-package "OMNI_MSGS"))
  (shadow 'OmniFeedback (find-package "OMNI_MSGS")))
(unless (find-package "OMNI_MSGS::OMNIFEEDBACK")
  (make-package "OMNI_MSGS::OMNIFEEDBACK"))

(in-package "ROS")
;;//! \htmlinclude OmniFeedback.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass omni_msgs::OmniFeedback
  :super ros::object
  :slots (_force _position ))

(defmethod omni_msgs::OmniFeedback
  (:init
   (&key
    ((:force __force) (instance geometry_msgs::Vector3 :init))
    ((:position __position) (instance geometry_msgs::Vector3 :init))
    )
   (send-super :init)
   (setq _force __force)
   (setq _position __position)
   self)
  (:force
   (&rest __force)
   (if (keywordp (car __force))
       (send* _force __force)
     (progn
       (if __force (setq _force (car __force)))
       _force)))
  (:position
   (&rest __position)
   (if (keywordp (car __position))
       (send* _position __position)
     (progn
       (if __position (setq _position (car __position)))
       _position)))
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Vector3 _force
    (send _force :serialization-length)
    ;; geometry_msgs/Vector3 _position
    (send _position :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Vector3 _force
       (send _force :serialize s)
     ;; geometry_msgs/Vector3 _position
       (send _position :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Vector3 _force
     (send _force :deserialize buf ptr-) (incf ptr- (send _force :serialization-length))
   ;; geometry_msgs/Vector3 _position
     (send _position :deserialize buf ptr-) (incf ptr- (send _position :serialization-length))
   ;;
   self)
  )

(setf (get omni_msgs::OmniFeedback :md5sum-) "e9083ac4fd95494e94fbb0c0f90b6c00")
(setf (get omni_msgs::OmniFeedback :datatype-) "omni_msgs/OmniFeedback")
(setf (get omni_msgs::OmniFeedback :definition-)
      "# This is the force as estimated from the applied torques as well as the
# current end effector position of the robot arm
geometry_msgs/Vector3  force
geometry_msgs/Vector3  position

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
float64 z
")



(provide :omni_msgs/OmniFeedback "e9083ac4fd95494e94fbb0c0f90b6c00")


