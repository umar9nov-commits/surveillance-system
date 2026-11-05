"""Read-only, low-rate camera preview over stdout; no ROS publishers."""
import argparse
import base64
import json
import os
import signal
import threading
import time

TOPICS={'external':'/surveillance/observer/image_raw','robot':'/surveillance/robot/image_raw'}
PREFIX='@@SURVEILLANCE_CAMERA@@'
PERIOD=2.0
MAX_AGE=4.0

def preview_size(width,height):
    if not 0<width<=8192 or not 0<height<=8192:raise ValueError('Invalid image dimensions')
    scale=min(1.,320/width,240/height)
    return max(1,round(width*scale)),max(1,round(height*scale))

class FrameGate:
    def __init__(self):self.latest=None;self.arrival=0.;self.sent=None
    def receive(self,message,now):self.latest=message;self.arrival=now
    def take(self,now):
        m=self.latest
        if m is None or now-self.arrival>MAX_AGE:return None
        stamp=(m.header.stamp.sec,m.header.stamp.nanosec)
        if stamp==(0,0) or stamp==self.sent:return None
        self.sent=stamp;return m

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',choices=tuple(TOPICS),default='external')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    import cv2
    from cv_bridge import CvBridge
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from rclpy.qos import QoSProfile,ReliabilityPolicy,HistoryPolicy
    from sensor_msgs.msg import Image
    cv2.setNumThreads(1)
    if args.check:print('Camera preview dependencies available.');return
    stop=threading.Event()
    def commands():
        import sys
        for line in sys.stdin:
            if line.strip()=='stop':break
        stop.set()
    threading.Thread(target=commands,daemon=True).start()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node=rclpy.create_node('flutter_camera_preview_'+args.source+'_'+str(os.getpid()))
    gate=FrameGate();bridge=CvBridge()
    def receive(message):gate.receive(message,time.monotonic())
    subscription=node.create_subscription(Image,TOPICS[args.source],receive,
        QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST))
    next_frame=0.
    try:
        while rclpy.ok() and not stop.is_set():
            rclpy.spin_once(node,timeout_sec=.1)
            now=time.monotonic()
            if now<next_frame:continue
            message=gate.take(now)
            if message is None:continue
            next_frame=now+PERIOD
            try:
                frame=bridge.imgmsg_to_cv2(message,desired_encoding='bgr8')
                size=preview_size(message.width,message.height)
                if (frame.shape[1],frame.shape[0])!=size:frame=cv2.resize(frame,size,interpolation=cv2.INTER_AREA)
                ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,55])
                if not ok:continue
                if time.monotonic()-gate.arrival>MAX_AGE:continue
                print(PREFIX+json.dumps(dict(kind='frame',source=args.source,jpeg=base64.b64encode(encoded).decode('ascii')),separators=(',',':')),flush=True)
            except BrokenPipeError:break
            except Exception as error:
                node.get_logger().warning('Preview frame skipped: '+str(error))
    except KeyboardInterrupt:pass
    finally:
        node.destroy_subscription(subscription);node.destroy_node();rclpy.try_shutdown()
if __name__=='__main__':main()
