import sys, h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
def load_h5(path):
    with h5py.File(path, "r") as f:
        kps = f["keypoints"][()]                 # (T, 5, 5, 3)
        ts  = f["timestamps"][()] if "timestamps" in f else np.arange(kps.shape[0], dtype=float)
    return kps, ts
def finger_edges():
    edges = []
    # 每指内部
    for f in range(5):
        base = f*5
        for j in range(4):
            edges.append((base+j, base+j+1))
    wrist_flat = 0            # (finger=0, joint=0)
    bases = [0, 5, 10, 15, 20]
    for b in bases[1:]:
        edges.append((wrist_flat, b))
    return edges
def to_video(h5_path, out_mp4, fps=15, flipz=False):
    kps, ts = load_h5(h5_path)
    if flipz:
        kps = kps.copy(); kps[:,:,2] *= -1
    T = kps.shape[0]
    pts = kps.reshape(T, -1, 3)         # (T, 25, 3)
    wrist_traj = kps[:, 0, 0, :]
    E = finger_edges()
    xyz = pts.reshape(-1, 3)
    mins = xyz.min(0); maxs = xyz.max(0)
    span = np.maximum(maxs - mins, 1e-6)
    mins -= 0.02 * span; maxs += 0.02 * span
    fig = plt.figure(figsize=(6.4, 6.4), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect([1,1,1])
    ax.set_title("Manus hand skeleton & wrist trajectory")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    with writer.saving(fig, out_mp4, dpi=120):
        for t in range(len(pts)):
            print(t, len(pts))
            p = pts[t]  # (25,3)
            ax.cla()
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])
            ax.set_box_aspect([1,1,1])
            ax.set_title(f"Manus skeleton — frame {t}/{T-1}  time={ts[t]:.3f}s")
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            for i, j in E:
                a, b = p[i], p[j]
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])
            ax.scatter(p[:,0], p[:,1], p[:,2], s=8)
            tr = wrist_traj[:t+1]
            ax.plot(tr[:,0], tr[:,1], tr[:,2], 'm-', lw=2)
            writer.grab_frame()
    print(f"[ok] wrote video -> {out_mp4}")
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python manus_video_traj.py <manus_data.h5> <out_1.mp4> [--fps 15] [--flipz]")
        '''
        python visual_manus.py /home/jolia/vr-hand-tracking/Franka-Teach/RUKA/data/right_hand/demonstration_right_four_fingers/manus_data.h5 out_haha.mp4 --fps 15 --flipz
        '''     
        sys.exit(1)
    h5_path = sys.argv[1]
    out_mp4 = sys.argv[2]
    fps = 15
    flipz = False
    for i, a in enumerate(sys.argv[3:]):
        if a == "--fps" and i+1 < len(sys.argv[3:]): fps = int(sys.argv[3:][i+1])
        if a == "--flipz": flipz = True
    to_video(h5_path, out_mp4, fps=fps, flipz=flipz)
    
    
        