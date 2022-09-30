;; Auto-generated. Do not edit!


(when (boundp 'omni_msgs::OmniState)
  (if (not (find-package "OMNI_MSGS"))
    (make-package "OMNI_MSGS"))
  (shadow 'OmniState (find-package "OMNI_MSGS")))
(unless (find-package "OMNI_MSGS::OMNISTATE")
  (make-package "OMNI_MSGS::OMNISTATE"))

(in-package "ROS")
;;//! \htmlinclude OmniState.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass omni_msgs::OmniState
  :super ros::object
  :slots (_header _locked _close_gripper _pose _current _velocity ))

(defmethod omni_msgs::OmniState
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:locked __locked) nil)
    ((:close_gripper __close_gripper) nil)
    ((:pose __pose) (instance geometry_msgs::Pose :init))
    ((:current __current) (instance geometry_msgs::Vector3 :init))
    ((:velocity __velocity) (instance geometry_msgs::Vector3 :init))
    )
   (send-super :init)
   (setq _header __header)
   (setq _locked __locked)
   (setq _close_gripper __close_gripper)
   (setq _pose __pose)
   (setq _current __current)
   (setq _velocity __velocity)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:locked
   (&optional (__locked :null))
   (if (not (eq __locked :null)) (setq _locked __locked)) _locked)
  (:close_gripper
   (&optional (__close_gripper :null))
   (if (not (eq __close_gripper :null)) (setq _close_gripper __close_gripper)) _close_gripper)
  (:pose
   (&rest __pose)
   (if (keywordp (car __pose))
       (send* _pose __pose)
     (progn
       (if __pose (setq _pose (car __pose)))
       _pose)))
  (:current
   (&rest __current)
   (if (keywordp (car __current))
       (send* _current __current)
     (progn
       (if __current (setq _current (car __current)))
       _current)))
  (:velocity
   (&rest __velocity)
   (if (keywordp (car __velocity))
       (send* _velocity __velocity)
     (progn
       (if __velocity (setq _velocity (car __velocity)))
       _velocity)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; bool _locked
    1
    ;; bool _close_gripper
    1
    ;; geometry_msgs/Pose _pose
    (send _pose :serialization-length)
    ;; geometry_msgs/Vector3 _current
    (send _current :serialization-length)
    ;; geometry_msgs/Vector3 _velocity
    (send _velocity :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; bool _locked
       (if _locked (write-byte -1 s) (write-byte 0 s))
     ;; bool _close_gripper
       (if _close_gripper (write-byte -1 s) (write-byte 0 s))
     ;; geometry_msgs/Pose _pose
       (send _pose :serialize s)
     ;; geometry_msgs/Vector3 _current
       (send _current :serialize s)
     ;; geometry_msgs/Vector3 _velocity
       (send _velocity :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; bool _locked
     (setq _locked (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _close_gripper
     (setq _close_gripper (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; geometry_msgs/Pose _pose
     (send _pose :deserialize buf ptr-) (incf ptr- (send _pose :serialization-length))
   ;; geometry_msgs/Vector3 _current
     (send _current :deserialize buf ptr-) (incf ptr- (send _current :serialization-length))
   ;; geometry_msgs/Vector3 _velocity
     (send _velocity :deserialize buf ptr-) (incf ptr- (send _velocity :serialization-length))
   ;;
   self)
  )

(setf (get omni_msgs::OmniState :md5sum-) "89c2a741de66e9e904f59a02b171dd6e")
(setf (get omni_msgs::OmniState :datatype-) "omni_msgs/OmniState")
(setf (get omni_msgs::OmniState :definition-)
      "std_msgs/Header         header

bool                    locked

bool                    close_gripper

geometry_msgs/Pose      pose        # meters

geometry_msgs/Vector3   current     # Amperes

geometry_msgs/Vector3   velocity    # meters/s

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

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



(provide :omni_msgs/OmniState "89c2a741de66e9e904f59a02b171dd6e")


