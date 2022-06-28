import cv2
import numpy as np

from utils import common
from planar.tracker import Tracker

vtx = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                  [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1],
                  [0, 0.5, 2], [1, 0.5, 2]])
ed = [(0, 1), (1, 2), (2, 3), (3, 0),
      (4, 5), (5, 6), (6, 7), (7, 4),
      (0, 4), (1, 5), (2, 6), (3, 7),
      (4, 8), (5, 8), (6, 9), (7, 9), (8, 9)]


def put_solid(vis, tracked):
    x0, y0, x1, y1 = tracked.target.rect
    quad = np.float32([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])
    h, w = vis.shape[:2]
    k = np.float64([[w, 0, 0.5 * (w - 1)],
                    [0, w, 0.5 * (h - 1)],
                    [0.0, 0.0, 1.0]])
    dist = np.zeros(4)
    _ret, rot, trans = cv2.solvePnP(quad, tracked.quad, k, dist)
    vx = vtx * [(x1 - x0), (y1 - y0), -(x1 - x0) * 0.3] + (x0, y0, 0)
    vx = cv2.projectPoints(vx, rot, trans, k, dist)[0].reshape(-1, 2)
    for i, j in ed:
        (x0, y0), (x1, y1) = vx[i], vx[j]
        cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 0), 2)


class App:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(10, 30)
        self.frame = None
        self.paused = False
        self.tracker = Tracker()

        cv2.namedWindow('AR Surface')
        self.sel = common.RectSelector('AR Surface', self.on_rect)

    def on_rect(self, rect):
        self.tracker.new(self.frame, rect)

    def run(self):
        while True:
            playing = not self.paused and not self.sel.dragging
            if playing or self.frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame = frame.copy()

            vis = self.frame.copy()
            if playing:
                tracked = self.tracker.track(self.frame)
                for tr in tracked:
                    cv2.polylines(vis, [np.int32(tr.quad)], True, (255, 255, 255), 2)
                    for (x, y) in np.int32(tr.p1):
                        cv2.circle(vis, (x, y), 2, (255, 255, 255))
                    put_solid(vis, tr)

            self.sel.draw(vis)
            cv2.imshow('AR Surface', vis)
            ch = cv2.waitKey(1)
            if ch == ord('c'):
                self.tracker.clear()
            if ch == 27:
                break


App(0).run()
