# zmq_sniff_keypoints.py
import zmq, sys, time, json
from ruka_hand.utils.constants import HOST, LEFT_STREAM_PORT  # 用你项目里的默认
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=HOST, help="publisher host (默认取项目常量)")
    ap.add_argument("--port", type=int, default=LEFT_STREAM_PORT, help="publisher port (默认取项目常量)")
    ap.add_argument("--topic", default="keypoints", help='订阅主题前缀；用 "" 订阅全部')
    ap.add_argument("--max", type=int, default=5, help="最多打印多少条后退出")
    ap.add_argument("--timeout", type=int, default=2000, help="poll 超时(ms)")
    args = ap.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM, 1000)

    # 订阅主题：为空字符串就订阅所有，便于探测真实主题名
    if args.topic == "":
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        topic_desc = "<ALL>"
    else:
        # 两种写法任选其一——这里用 bytes 更稳
        sock.setsockopt(zmq.SUBSCRIBE, args.topic.encode("utf-8"))
        # 或：sock.setsockopt_string(zmq.SUBSCRIBE, args.topic)
        topic_desc = args.topic

    endpoint = f"tcp://{args.host}:{args.port}"
    sock.connect(endpoint)

    print(f"[sniff] SUB connect {endpoint} topic={topic_desc}")

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    # 给 PUB/SUB 建立时间；避免“迟到订阅者”丢首包
    time.sleep(0.3)

    got = 0
    while True:
        socks = dict(poller.poll(args.timeout))
        if sock in socks:
            frames = sock.recv_multipart()  # 常见格式：[topic, payload]
            try:
                topic = frames[0].decode("utf-8", errors="ignore")
            except Exception:
                topic = repr(frames[0])
            payload = frames[-1]  # 载荷通常在最后一帧

            preview = payload[:120]
            # 尝试 JSON 预览
            text_preview = None
            try:
                text_preview = json.dumps(json.loads(payload.decode("utf-8")), ensure_ascii=False)[:120]
            except Exception:
                pass

            print(f"[sniff] frames={len(frames)} topic={topic!r} sizes={[len(f) for f in frames]}")
            if text_preview:
                print(f"[sniff] json: {text_preview} ...")
            else:
                print(f"[sniff] head-bytes: {preview!r} ...")

            got += 1
            if got >= args.max:
                break
        else:
            print("[sniff] no message in timeout…可能没在发，或端口/主题不对（可用 --topic '' 订阅全部再看）")

if __name__ == "__main__":
    main()
