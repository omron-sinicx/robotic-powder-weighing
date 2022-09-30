; Auto-generated. Do not edit!


(cl:in-package omni_msgs-msg)


;//! \htmlinclude OmniButtonEvent.msg.html

(cl:defclass <OmniButtonEvent> (roslisp-msg-protocol:ros-message)
  ((grey_button
    :reader grey_button
    :initarg :grey_button
    :type cl:integer
    :initform 0)
   (white_button
    :reader white_button
    :initarg :white_button
    :type cl:integer
    :initform 0))
)

(cl:defclass OmniButtonEvent (<OmniButtonEvent>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <OmniButtonEvent>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'OmniButtonEvent)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name omni_msgs-msg:<OmniButtonEvent> is deprecated: use omni_msgs-msg:OmniButtonEvent instead.")))

(cl:ensure-generic-function 'grey_button-val :lambda-list '(m))
(cl:defmethod grey_button-val ((m <OmniButtonEvent>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader omni_msgs-msg:grey_button-val is deprecated.  Use omni_msgs-msg:grey_button instead.")
  (grey_button m))

(cl:ensure-generic-function 'white_button-val :lambda-list '(m))
(cl:defmethod white_button-val ((m <OmniButtonEvent>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader omni_msgs-msg:white_button-val is deprecated.  Use omni_msgs-msg:white_button instead.")
  (white_button m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <OmniButtonEvent>) ostream)
  "Serializes a message object of type '<OmniButtonEvent>"
  (cl:let* ((signed (cl:slot-value msg 'grey_button)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'white_button)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <OmniButtonEvent>) istream)
  "Deserializes a message object of type '<OmniButtonEvent>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'grey_button) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'white_button) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<OmniButtonEvent>)))
  "Returns string type for a message object of type '<OmniButtonEvent>"
  "omni_msgs/OmniButtonEvent")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'OmniButtonEvent)))
  "Returns string type for a message object of type 'OmniButtonEvent"
  "omni_msgs/OmniButtonEvent")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<OmniButtonEvent>)))
  "Returns md5sum for a message object of type '<OmniButtonEvent>"
  "fb77877e6b639935c3360838062f05f0")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'OmniButtonEvent)))
  "Returns md5sum for a message object of type 'OmniButtonEvent"
  "fb77877e6b639935c3360838062f05f0")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<OmniButtonEvent>)))
  "Returns full string definition for message of type '<OmniButtonEvent>"
  (cl:format cl:nil "int32 grey_button~%int32 white_button~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'OmniButtonEvent)))
  "Returns full string definition for message of type 'OmniButtonEvent"
  (cl:format cl:nil "int32 grey_button~%int32 white_button~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <OmniButtonEvent>))
  (cl:+ 0
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <OmniButtonEvent>))
  "Converts a ROS message object to a list"
  (cl:list 'OmniButtonEvent
    (cl:cons ':grey_button (grey_button msg))
    (cl:cons ':white_button (white_button msg))
))
