from collections import namedtuple

import cv2
import numpy as np

flann = dict(algorithm=6, table_number=6,
             key_size=12, multi_probe_level=1)

THRESH = 10

Surface = namedtuple('Surface', 'image, rect, pt, sdc, data')
Tracked = namedtuple('Tracked', 'target, p0, p1, H, quad')


class Tracker:
    def __init__(self):
        self.detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.FlannBasedMatcher(flann, {})
        self.targets = []
        self.f_points = []

    def new(self, image, rect, data=None):
        x0, y0, x1, y1 = rect
        raw_points, raw_sdc = self.detect_features(image)
        points, sdc = [], []
        for kp, desc in zip(raw_points, raw_sdc):
            x, y = kp.pt
            if x0 <= x <= x1 and y0 <= y <= y1:
                points.append(kp)
                sdc.append(desc)
        sdc = np.uint8(sdc)
        self.matcher.add([sdc])
        target = Surface(image=image, rect=rect, pt=points, sdc=sdc, data=data)
        self.targets.append(target)

    def clear(self):
        self.targets = []
        self.matcher.clear()

    def track(self, frame):
        self.f_points, sdc = self.detect_features(frame)
        if len(self.f_points) < THRESH:
            return []
        matches = self.matcher.knnMatch(sdc, k=2)
        matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
        if len(matches) < THRESH:
            return []
        mt = [[] for _ in range(len(self.targets))]
        for m in matches:
            mt[m.imgIdx].append(m)
        tracked = []
        for ix, matches in enumerate(mt):
            if len(matches) < THRESH:
                continue
            target = self.targets[ix]
            p0 = [target.pt[m.trainIdx].pt for m in matches]
            p1 = [self.f_points[m.queryIdx].pt for m in matches]
            p0, p1 = np.float32((p0, p1))
            hg, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
            status = status.ravel() != 0
            if status.sum() < THRESH:
                continue
            p0, p1 = p0[status], p1[status]
            x0, y0, x1, y1 = target.rect
            quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), hg).reshape(-1, 2)
            tracked.append(Tracked(target=target, p0=p0, p1=p1, H=hg, quad=quad))
        tracked.sort(key=lambda t: len(t.p0), reverse=True)
        return tracked

    def detect_features(self, frame):
        sp, sdc = self.detector.detectAndCompute(frame, None)
        if sdc is None:
            sdc = []
        return sp, sdc
