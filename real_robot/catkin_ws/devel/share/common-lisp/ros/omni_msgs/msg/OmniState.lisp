; Auto-generated. Do not edit!


(cl:in-package omni_msgs-msg)


;//! \htmlinclude OmniState.msg.html

(cl:defclass <OmniState> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (locked
    :reader locked
    :initarg :locked
    :type cl:boolean
    :initform cl:nil)
   (close_gripper
    :reader close_gripper
    :initarg :close_gripper
    :type cl:boolean
    :initform cl:nil)
   (pose
    :reader pose
    :initarg :pose
    :type geometry_msgs-msg:Pose
    :initform (cl:make-instance 'geometry_msgs-msg:Pose))
   (current
    :reader current
    :initarg :current
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (velocity
    :reader velocity
    :initarg :velocity
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3)))
)

(cl:defclass OmniState (<OmniState>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <OmniState>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'OmniState)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name omni_msgs-msg:<OmniState> is deprecated: use omni_msgs-msg:OmniState instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <OmniState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader omni_msgs-msg:header-val is deprecated.  Use omni_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'locked-val :lambda-list '(m))
(cl:defmethod locked-val ((m <OmniState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader omni_msgs-msg:locked-val is deprecated.  Use omni_msgs-msg:locked instead.")
  (locked m))

(cl:ensure-generic-function 'close_gripper-val :lambda-list '(m))
(cl:defmethod close_gripper-val ((m <OmniState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader omni_msgs-msg:close_gripper-val is deprecated.  Use omni_msgs-msg:close_gripper instead.")
  (close_gripper m))

(cl:ensure-generic-function 'pose-val :lambda-list '(m))
(cl:defmethod pose-val ((m <OmniState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader omni_msgs-msg:pose-val is deprecated.  Use omni_msgs-msg:pose instead.")
  (pose m))

(cl:ensure-generic-function 'current-val :lambda-list '(m))
(cl:defmethod current-val ((m <OmniState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader omni_msgs-msg:current-val is deprecated.  Use omni_msgs-msg:current instead.")
  (current m))

(cl:ensure-generic-function 'velocity-val :lambda-list '(m))
(cl:defmethod velocity-val ((m <OmniState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader omni_msgs-msg:velocity-val is deprecated.  Use omni_msgs-msg:velocity instead.")
  (velocity m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <OmniState>) ostream)
  "Serializes a message object of type '<OmniState>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'locked) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'close_gripper) 1 0)) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'pose) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'current) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'velocity) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <OmniState>) istream)
  "Deserializes a message object of type '<OmniState>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:setf (cl:slot-value msg 'locked) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'close_gripper) (cl:not (cl:zerop (cl:read-byte istream))))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'pose) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'current) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'velocity) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<OmniState>)))
  "Returns string type for a message object of type '<OmniState>"
  "omni_msgs/OmniState")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'OmniState)))
  "Returns string type for a message object of type 'OmniState"
  "omni_msgs/OmniState")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<OmniState>)))
  "Returns md5sum for a message object of type '<OmniState>"
  "89c2a741de66e9e904f59a02b171dd6e")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'OmniState)))
  "Returns md5sum for a message object of type 'OmniState"
  "89c2a741de66e9e904f59a02b171dd6e")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<OmniState>)))
  "Returns full string definition for message of type '<OmniState>"
  (cl:format cl:nil "std_msgs/Header         header~%~%bool                    locked~%~%bool                    close_gripper~%~%geometry_msgs/Pose      pose        # meters~%~%geometry_msgs/Vector3   current     # Amperes~%~%geometry_msgs/Vector3   velocity    # meters/s~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'OmniState)))
  "Returns full string definition for message of type 'OmniState"
  (cl:format cl:nil "std_msgs/Header         header~%~%bool                    locked~%~%bool                    close_gripper~%~%geometry_msgs/Pose      pose        # meters~%~%geometry_msgs/Vector3   current     # Amperes~%~%geometry_msgs/Vector3   velocity    # meters/s~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <OmniState>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     1
     1
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'pose))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'current))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'velocity))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <OmniState>))
  "Converts a ROS message object to a list"
  (cl:list 'OmniState
    (cl:cons ':header (header msg))
    (cl:cons ':locked (locked msg))
    (cl:cons ':close_gripper (close_gripper msg))
    (cl:cons ':pose (pose msg))
    (cl:cons ':current (current msg))
    (cl:cons ':velocity (velocity msg))
))
