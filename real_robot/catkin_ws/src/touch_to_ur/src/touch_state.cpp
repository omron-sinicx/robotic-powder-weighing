#ifdef  _WIN64
#pragma warning (disable:4996)
#endif

#include <bits/stdc++.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <typeinfo>
#include <conio.h>

#include <HD/hd.h>
#include <HL/hl.h>
#include <HDU/hduMatrix.h>
#include <HDU/hduError.h>
#include <HLU/hlu.h>
#include <HDU/hduQuaternion.h>

#include <ros/ros.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>

geometry_msgs::Vector3 jointRotation;
geometry_msgs::Quaternion proxyRotation;
geometry_msgs::Point proxyPosition;

class TouchStatePublisher{
    public:
        TouchStatePublisher();
        void publish();
    private:
        ros::NodeHandle nh;
        ros::Publisher jointRotationPub, proxyRotationPub, proxyPositionPub;
};

TouchStatePublisher::TouchStatePublisher(){
    jointRotationPub = nh.advertise<geometry_msgs::Vector3>("/touch_state/joint_rotation", 1);
    proxyRotationPub = nh.advertise<geometry_msgs::Quaternion>("/touch_state/proxy_rotation", 1);
    proxyPositionPub = nh.advertise<geometry_msgs::Point>("/touch_state/proxy_position", 1);
}

void TouchStatePublisher::publish(){
    jointRotationPub.publish(jointRotation);
    proxyRotationPub.publish(proxyRotation);
    proxyPositionPub.publish(proxyPosition);
}

void HLCALLBACK motionCallback(HLenum event, HLuint object, HLenum thread, HLcache *cache, void *userdata){

    hduQuaternion proxyRotq;
    hduVector3Dd jointAngles, proxyPos;    
    hdGetDoublev(HD_CURRENT_JOINT_ANGLES,jointAngles);
    hlGetDoublev(HL_PROXY_ROTATION, proxyRotq);
    hlGetDoublev(HL_PROXY_POSITION, proxyPos);
    std::cout << "Joint Rotation: " << jointAngles << std::endl;
    std::cout << "Proxy Rotation: " << proxyRotq << std::endl;
    std::cout << "Proxy Position: " << proxyPos << std::endl;
    std::cout << "--Press any key to exit--" << std::endl << std::endl;
    //fprintf(stdout, "Moving... %f\n", jointAngles);

    std::string jointAngles_str = "", proxyRotq_str = "", proxyPos_str = "";
    for (int i=0;i<3;i++){
        // std::cout << typeid(jointAngles[i]).name() << std::endl;
        std::stringstream s1;
        s1<<jointAngles[i];
        std::string temp;
        s1>>temp;
        jointAngles_str += temp;
        if (i<2) jointAngles_str += " ";
    }

    for (int i=0;i<4;i++){
        // std::cout << typeid(proxyRotq[i]).name() << std::endl;
        std::stringstream s2;
        s2<<proxyRotq[i];
        std::string temp;
        s2>>temp;
        proxyRotq_str += temp;
        if (i<3) proxyRotq_str += " ";
    }

    for (int i=0;i<3;i++){
        // std::cout << typeid(proxyPos[i]).name() << std::endl;
        std::stringstream s3;
        s3<<proxyPos[i];
        std::string temp;
        s3>>temp;
        proxyPos_str += temp;
        if (i<2) proxyPos_str += " ";
    }

    jointRotation.x = jointAngles[0];
    jointRotation.y = jointAngles[1];
    jointRotation.z = jointAngles[2];
    
    proxyRotation.x = proxyRotq[0];
    proxyRotation.y = proxyRotq[1];
    proxyRotation.z = proxyRotq[2];
    proxyRotation.w = proxyRotq[3];

    proxyPosition.x = proxyPos[0];
    proxyPosition.y = proxyPos[1];
    proxyPosition.z = proxyPos[2];
}

int main(int argc, char** argv){
    ros::init(argc, argv, "touch_state_publisher");
    TouchStatePublisher touchStatePublisher;

    HHD hHD;
    HDErrorInfo error;
    HHLRC hHLRC;

	hHD = hdInitDevice(HD_DEFAULT_DEVICE);
    if (HD_DEVICE_ERROR(error = hdGetError())) 
    {
        hduPrintError(stderr, &error, "Failed to initialize haptic device");
        fprintf(stderr, "\nPress any key to quit.\n");
        getchar();
        exit(-1);
    }
    hdMakeCurrentDevice(hHD);    

    hHLRC = hlCreateContext(hHD);
    hlMakeCurrent(hHLRC);
    
    hlDisable(HL_USE_GL_MODELVIEW);

    hlAddEventCallback(HL_EVENT_MOTION, HL_OBJECT_ANY, HL_CLIENT_THREAD, 
                       &motionCallback, NULL);

    ros::Rate loopRate(100);
    while (ros::ok() && !_kbhit()){
        hlBeginFrame();
        hlCheckEvents();
        hlEndFrame();

        // Check for any errors.
        HLerror error;
        while (HL_ERROR(error = hlGetError()))
        {
            fprintf(stderr, "HL Error: %s\n", error.errorCode);
            
            if (error.errorCode == HL_DEVICE_ERROR)
            {
                hduPrintError(stderr, &error.errorInfo,
                    "Error during haptic rendering\n");
            }
        }
        
        touchStatePublisher.publish();
        ros::spinOnce();
        loopRate.sleep();
    }

}
