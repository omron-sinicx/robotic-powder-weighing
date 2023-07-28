// Auto-generated. Do not edit!

// (in-package omni_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class OmniButtonEvent {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.grey_button = null;
      this.white_button = null;
    }
    else {
      if (initObj.hasOwnProperty('grey_button')) {
        this.grey_button = initObj.grey_button
      }
      else {
        this.grey_button = 0;
      }
      if (initObj.hasOwnProperty('white_button')) {
        this.white_button = initObj.white_button
      }
      else {
        this.white_button = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type OmniButtonEvent
    // Serialize message field [grey_button]
    bufferOffset = _serializer.int32(obj.grey_button, buffer, bufferOffset);
    // Serialize message field [white_button]
    bufferOffset = _serializer.int32(obj.white_button, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type OmniButtonEvent
    let len;
    let data = new OmniButtonEvent(null);
    // Deserialize message field [grey_button]
    data.grey_button = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [white_button]
    data.white_button = _deserializer.int32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 8;
  }

  static datatype() {
    // Returns string type for a message object
    return 'omni_msgs/OmniButtonEvent';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'fb77877e6b639935c3360838062f05f0';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int32 grey_button
    int32 white_button
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new OmniButtonEvent(null);
    if (msg.grey_button !== undefined) {
      resolved.grey_button = msg.grey_button;
    }
    else {
      resolved.grey_button = 0
    }

    if (msg.white_button !== undefined) {
      resolved.white_button = msg.white_button;
    }
    else {
      resolved.white_button = 0
    }

    return resolved;
    }
};

module.exports = OmniButtonEvent;
