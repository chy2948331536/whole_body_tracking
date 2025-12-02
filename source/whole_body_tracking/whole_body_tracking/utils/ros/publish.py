#!/usr/bin/env python3
import argparse, random, time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time as RosTime
from sensor_msgs.msg import JointState

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

class OneFilePublisher(Node):
    def __init__(self, topic: str, size: int, prefix: str):
        # 检测 rclpy 是否已初始化
        if not rclpy.ok():
            rclpy.init()
        super().__init__('one_file_pub')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.pub = self.create_publisher(JointState, topic, qos)
        self.names = [f'{prefix}{i+1}' for i in range(size)]
        self.size = size

    def cb(self, data, stamp):
        # 只用 position 携带数据；header.stamp 供 PlotJuggler 做时间轴
        msg = JointState()
        if stamp is None:
            msg.header.stamp = self.get_clock().now().to_msg()
        elif isinstance(stamp, RosTime):
            msg.header.stamp = stamp.to_msg()
        elif hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
            # builtin_interfaces.msg.Time
            msg.header.stamp = stamp
        elif isinstance(stamp, (int, float)):
            msg.header.stamp = RosTime(seconds=float(stamp)).to_msg()
        elif isinstance(stamp, tuple) and len(stamp) == 2:
            sec, nsec = stamp
            msg.header.stamp = RosTime(seconds=float(stamp), nanoseconds=int(nsec)).to_msg()
        else:
            raise TypeError("Unsupported stamp type")

        # 处理 tensor 类型的数据，转换为 float 类型的 list
        if HAS_TORCH and isinstance(data, torch.Tensor):
            # 确保是 float 类型（处理 bool tensor）
            data = data.float().cpu().numpy().flatten().tolist()
        
        # 确保 list 中所有元素都是 float（处理可能的 bool 或其他类型）
        if isinstance(data, list):
            data = [float(x) for x in data]
        
        msg.name = self.names
        msg.position = data
        self.pub.publish(msg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', default='/joint_cmd_stamped')
    parser.add_argument('--hz', type=float, default=60.0)
    parser.add_argument('--size', type=int, default=12)
    parser.add_argument('--prefix', type=str, default='J', help='假名前缀')
    parser.add_argument('--torch', action='store_true', help='使用PyTorch生成数据')
    parser.add_argument('--stamp_mode', choices=['now', 'system', 'sim'], default='sim',
                        help='时间戳来源：now=ROS clock，system=time.time()，sim=按dt递增')
    parser.add_argument('--t0', type=float, default=0.0, help='sim 模式起始时间（秒）')
    args = parser.parse_args()

    node = OneFilePublisher(args.topic, args.size, args.prefix)
    use_torch = args.torch and HAS_TORCH
    dt = 1.0 / args.hz
    k = 0
    t0_sys = time.time()

    try:
        while rclpy.ok():
            if use_torch:
                data = torch.randn(args.size, dtype=torch.float32).tolist()
            else:
                data = [random.random() for _ in range(args.size)]
            
            # 选择时间戳
            if args.stamp_mode == 'now':
                stamp = None  # 由 publish 内部使用 node clock now()
            elif args.stamp_mode == 'system':
                # 直接用系统时间（墙钟秒）
                stamp = time.time()
            else:  # 'sim'
                # 离散仿真时间：t = t0 + k*dt
                stamp = args.t0 + k * dt
                k += 1

            node.cb(data, stamp=stamp)
            # 处理事件队列，避免阻塞
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(dt)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
