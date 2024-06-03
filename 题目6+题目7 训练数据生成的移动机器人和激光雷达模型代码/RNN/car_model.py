# 依赖库 opencv-python

import cv2
import numpy as np


def _rot_pos(x, y, phi_):   #旋转后的坐标，x,y为原坐标，phi_为旋转角度
    phi = np.deg2rad(phi_)
    return np.array((x * np.cos(phi) + y * np.sin(phi), -x * np.sin(phi) + y * np.cos(phi)))


def _draw_rectangle(img, x, y, u, v, phi, color=(0, 0, 0), size=1):  #画矩形, x,y为中心坐标，u,v为长和宽，phi为旋转角度
    pts1 = _rot_pos(-u / 2, -v / 2, phi) + np.array((x, y))
    pts2 = _rot_pos(u / 2, -v / 2, phi) + np.array((x, y))
    pts3 = _rot_pos(-u / 2, v / 2, phi) + np.array((x, y)) 
    pts4 = _rot_pos(u / 2, v / 2, phi) + np.array((x, y))
    cv2.line(img, tuple(pts1.astype(int).tolist()), tuple(pts2.astype(int).tolist()), color, size)
    cv2.line(img, tuple(pts1.astype(int).tolist()), tuple(pts3.astype(int).tolist()), color, size)
    cv2.line(img, tuple(pts3.astype(int).tolist()), tuple(pts4.astype(int).tolist()), color, size)
    cv2.line(img, tuple(pts2.astype(int).tolist()), tuple(pts4.astype(int).tolist()), color, size)
    return img


class KinematicModel:   #运动学模型
    def __init__(self,
                 v_range=60,  #速度范围
                 w_range=45,    #角速度范围

                 d=14,  #轮距
                 # Wheel Size
                 wu=10, #轮长
                 wv=4,  #轮宽
                 # Car Size
                 car_w=24,  #车宽
                 car_f=20,  #车头长
                 car_r=10,  #车尾长
                 dt=0.1
                 ):
        # Rear Wheel as Origin Point
        # Initialize State
        self.init_state((0, 0, 0))  #初始化状态

        # ============ Car Parameter ============
        # Control Constrain
        self.v_range = v_range
        self.w_range = w_range
        # Wheel Distance
        self.d = d
        # Wheel size
        self.wu = wu
        self.wv = wv
        # Car size
        self.car_w = car_w
        self.car_f = car_f
        self.car_r = car_r
        self._compute_car_box()

        self.dt = dt    #时间间隔

    def init_state(self, pos):
        self.x = pos[0]
        self.y = pos[1] 
        self.yaw = pos[2]   #偏航角
        self.v = 0  #速度
        self.a = 0  #加速度
        self.w = 0  #角速度
        self.record = []        #记录

    def update(self):
        # Control Constrain
        if self.v > self.v_range:
            self.v = self.v_range
        elif self.v < -self.v_range:
            self.v = -self.v_range
        if self.w > self.w_range:
            self.w = self.w_range
        elif self.w < -self.w_range:
            self.w = -self.w_range

        # Motion    
        self.x += self.v * np.cos(np.deg2rad(self.yaw)) * self.dt
        self.y += self.v * np.sin(np.deg2rad(self.yaw)) * self.dt
        self.yaw += self.w * self.dt
        self.yaw = self.yaw % 360
        self.record.append((self.x, self.y, self.yaw))  #记录
        self._compute_car_box() #计算车辆的边界框

    def redo(self):
        self.x -= self.v * np.cos(np.deg2rad(self.yaw)) * self.dt
        self.y -= self.v * np.sin(np.deg2rad(self.yaw)) * self.dt
        self.yaw -= self.w * self.dt
        self.yaw = self.yaw % 360
        self.record.pop()   #弹出记录

    def control(self, v, w):
        self.v = v
        self.w = w

        # Control Constrain
        if self.v > self.v_range:
            self.v = self.v_range
        elif self.v < -self.v_range:
            self.v = -self.v_range
        if self.w > self.w_range:
            self.w = self.w_range
        elif self.w < -self.w_range:
            self.w = -self.w_range

    def state_str(self):    #状态字符串
        return "x={:.4f}, y={:.4f}, v={:.4f}, a={:.4f}, yaw={:.4f}, w={:.4f}".format(self.x, self.y, self.v, self.a,
                                                                                     self.yaw, self.w)

    def _compute_car_box(self): #计算车辆的边界框
        pts1 = _rot_pos(self.car_f, self.car_w / 2, -self.yaw) + np.array((self.x, self.y))
        pts2 = _rot_pos(self.car_f, -self.car_w / 2, -self.yaw) + np.array((self.x, self.y))
        pts3 = _rot_pos(-self.car_r, self.car_w / 2, -self.yaw) + np.array((self.x, self.y))
        pts4 = _rot_pos(-self.car_r, -self.car_w / 2, -self.yaw) + np.array((self.x, self.y))
        self.car_box = (pts1.astype(int), pts2.astype(int), pts3.astype(int), pts4.astype(int))

    def render(self, img=np.ones((600, 600, 3))):
        # ######### Draw History ##########
        rec_max = 1000
        start = 0 if len(self.record) < rec_max else len(self.record) - rec_max
        # Draw Trajectory
        color = (0 / 255, 97 / 255, 255 / 255)
        for i in range(start, len(self.record) - 1):    #画轨迹
            cv2.line(img, (int(self.record[i][0]), int(self.record[i][1])),
                     (int(self.record[i + 1][0]), int(self.record[i + 1][1])), color, 1)

        # ######### Draw Car ##########
        # Car box
        pts1, pts2, pts3, pts4 = self.car_box
        color = (0, 0, 0)
        size = 1
        cv2.line(img, tuple(pts1.astype(int).tolist()), tuple(pts2.astype(int).tolist()), color, size)
        cv2.line(img, tuple(pts1.astype(int).tolist()), tuple(pts3.astype(int).tolist()), color, size)
        cv2.line(img, tuple(pts3.astype(int).tolist()), tuple(pts4.astype(int).tolist()), color, size)
        cv2.line(img, tuple(pts2.astype(int).tolist()), tuple(pts4.astype(int).tolist()), color, size)
        # Car center & direction
        t1 = _rot_pos(6, 0, -self.yaw) + np.array((self.x, self.y))
        t2 = _rot_pos(0, 4, -self.yaw) + np.array((self.x, self.y))
        t3 = _rot_pos(0, -4, -self.yaw) + np.array((self.x, self.y))
        cv2.line(img, (int(self.x), int(self.y)), (int(t1[0]), int(t1[1])), (0, 0, 1), 2)
        cv2.line(img, (int(t2[0]), int(t2[1])), (int(t3[0]), int(t3[1])), (1, 0, 0), 2)

        # ######### Draw Wheels ##########
        w1 = _rot_pos(0, self.d, -self.yaw) + np.array((self.x, self.y))
        w2 = _rot_pos(0, -self.d, -self.yaw) + np.array((self.x, self.y))
        # 4 Wheels
        img = _draw_rectangle(img, int(w1[0]), int(w1[1]), self.wu, self.wv, -self.yaw)    # 旋转后坐标以及长和宽
        img = _draw_rectangle(img, int(w2[0]), int(w2[1]), self.wu, self.wv, -self.yaw)
        # Axle
        img = cv2.line(img, tuple(w1.astype(int).tolist()), tuple(w2.astype(int).tolist()), (0, 0, 0), 1)
        return img


# ================= main =================
if __name__ == "__main__":
    car = KinematicModel()
    car.init_state((300, 300, 0))
    while True:
        print(
            "\rx={}, y={}, v={}, yaw={}, w={}".format(str(car.x)[:5], str(car.y)[:5], str(car.v)[:5], str(car.yaw)[:5],
                                                      str(car.w)[:5]), end="\t")
        img = np.ones((600, 600, 3))
        car.update()
        img = car.render(img)
        img = cv2.flip(img, 0)
        cv2.imshow("demo", img)
        k = cv2.waitKey(1)
        if k == ord("a"):
            car.w += 5
        elif k == ord("d"):
            car.w -= 5
        elif k == ord("w"):
            car.v += 4
        elif k == ord("s"):
            car.v -= 4
        elif k == 27:
            print()
            break
