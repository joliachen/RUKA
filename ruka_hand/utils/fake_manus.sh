python3 - <<'PY'
import time, zmq
ctx = zmq.Context()
s = ctx.socket(zmq.PUB)
s.bind("tcp://*:5050")              # 注意：对外可达，别写 127.0.0.1
time.sleep(0.5)                     # 等订阅建立
i = 0
while True:
    s.send_multipart([b"keypoints", f"hello {i}".encode()])
    i += 1
    time.sleep(0.2)
PY
