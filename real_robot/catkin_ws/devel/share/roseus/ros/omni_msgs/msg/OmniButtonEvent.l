;; Auto-generated. Do not edit!


(when (boundp 'omni_msgs::OmniButtonEvent)
  (if (not (find-package "OMNI_MSGS"))
    (make-package "OMNI_MSGS"))
  (shadow 'OmniButtonEvent (find-package "OMNI_MSGS")))
(unless (find-package "OMNI_MSGS::OMNIBUTTONEVENT")
  (make-package "OMNI_MSGS::OMNIBUTTONEVENT"))

(in-package "ROS")
;;//! \htmlinclude OmniButtonEvent.msg.html


(defclass omni_msgs::OmniButtonEvent
  :super ros::object
  :slots (_grey_button _white_button ))

(defmethod omni_msgs::OmniButtonEvent
  (:init
   (&key
    ((:grey_button __grey_button) 0)
    ((:white_button __white_button) 0)
    )
   (send-super :init)
   (setq _grey_button (round __grey_button))
   (setq _white_button (round __white_button))
   self)
  (:grey_button
   (&optional __grey_button)
   (if __grey_button (setq _grey_button __grey_button)) _grey_button)
  (:white_button
   (&optional __white_button)
   (if __white_button (setq _white_button __white_button)) _white_button)
  (:serialization-length
   ()
   (+
    ;; int32 _grey_button
    4
    ;; int32 _white_button
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int32 _grey_button
       (write-long _grey_button s)
     ;; int32 _white_button
       (write-long _white_button s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int32 _grey_button
     (setq _grey_button (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _white_button
     (setq _white_button (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get omni_msgs::OmniButtonEvent :md5sum-) "fb77877e6b639935c3360838062f05f0")
(setf (get omni_msgs::OmniButtonEvent :datatype-) "omni_msgs/OmniButtonEvent")
(setf (get omni_msgs::OmniButtonEvent :definition-)
      "int32 grey_button
int32 white_button

")



(provide :omni_msgs/OmniButtonEvent "fb77877e6b639935c3360838062f05f0")


