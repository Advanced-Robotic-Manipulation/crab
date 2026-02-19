#!/home/aibek/miniconda3/envs/mobile-robot/bin/python
import zmq
import sys

ORIN_IP = "192.168.50.239"
ZMQ_PORT = 5556

try:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    sock.connect(f"tcp://{ORIN_IP}:{ZMQ_PORT}")
    data = sock.recv()
    sock.close()
    ctx.term()
    sys.exit(0)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
