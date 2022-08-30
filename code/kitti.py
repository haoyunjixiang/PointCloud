import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import sys

sys.path.append('../../')

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.utils import get_angle_from_box3d, check_range
from lib.datasets.kitti_utils import get_objects_from_label
from lib.datasets.kitti_utils import Calibration
from lib.datasets.kitti_utils import get_affine_transform
from lib.datasets.kitti_utils import affine_transform
from lib.datasets.kitti_utils import compute_box_3d
import pdb
import time

import numpy as np
import os
import yaml
from pyquaternion import Quaternion
import math

def get_mean_size(root_path,class_names):
    # root_path = "E:\Learn\data\\3D\\ji\\"
    names_file = open(os.path.join(root_path,"sample.txt"))
    dict_sum = {}
    dict_num = {}
    for line in names_file:
        txt_file = open(os.path.join(root_path,"label",line.strip()+".txt"))
        for line in txt_file:
            line = line.strip()
            # print(line)
            arr = line.split(" ")
            if arr[0] in dict_sum.keys() :
                if float(arr[8]) > 0.00001:
                    dict_sum[arr[0]] = [dict_sum[arr[0]][0] + float(arr[8]),dict_sum[arr[0]][1] + float(arr[9]),dict_sum[arr[0]][2] + float(arr[10])]
                    dict_num[arr[0]] = dict_num[arr[0]] + 1
            else:
                if float(arr[8]) > 0.00001:
                    dict_sum[arr[0]] = [float(arr[8]), float(arr[9]),float(arr[10])]
                    dict_num[arr[0]] = 1
    for key in dict_sum.keys():
        dimension = np.asarray(dict_sum[key],dtype=float)
        dict_sum[key] = dimension / dict_num[key]

    mean_size = []
    for cls in class_names:
        mean_size.append(dict_sum[cls])

    return np.asarray(mean_size)

class Data:
    """ class Data """
    def __init__(self, obj_type="unset", truncation=-1, occlusion=-1, \
                 obs_angle=-10, x1=-1, y1=-1, x2=-1, y2=-1, w=-1, h=-1, l=-1, \
                 X=-1000, Y=-1000, Z=-1000, yaw=-10, score=-1000, detect_id=-1, \
                 vx=0, vy=0, vz=0):
        """init object data"""
        self.obj_type = obj_type
        self.truncation = truncation
        self.occlusion = occlusion
        self.obs_angle = obs_angle
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.w = w
        self.h = h
        self.l = l
        self.X = X
        self.Y = Y
        self.Z = Z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.yaw = yaw
        self.score = score
        self.ignored = False
        self.valid = False
        self.detect_id = detect_id

    def __str__(self):
        """ str """
        attrs = vars(self)
        return '\n'.join("%s: %s" % item for item in attrs.items())


def read_kitti_ext(extfile):
    """read extrin"""
    text_file = open(extfile, 'r')
    cont = text_file.read()
    x = yaml.safe_load(cont)
    r = x['transform']['rotation']
    t = x['transform']['translation']
    q = Quaternion(r['w'], r['x'], r['y'], r['z'])
    m = q.rotation_matrix
    m = np.matrix(m).reshape((3, 3))
    t = np.matrix([t['x'], t['y'], t['z']]).T
    p1 = np.vstack((np.hstack((m, t)), np.array([0, 0, 0, 1])))
    return np.array(p1.I)


def read_kitti_cal(calfile):
    """
    Reads the kitti calibration projection matrix (p2) file from disc.
    Args:
        calfile (str): path to single calibration file
    """
    text_file = open(calfile, 'r')
    for line in text_file:
        parsed = line.split('\n')[0].split(' ')
        # bbGt annotation in text format of:
        # cls x y w h occ x y w h ign ang
        if parsed is not None and parsed[0] == 'P2:':
            p2 = np.zeros([4, 4], dtype=float)
            p2[0, 0] = parsed[1]
            p2[0, 1] = parsed[2]
            p2[0, 2] = parsed[3]
            p2[0, 3] = parsed[4]
            p2[1, 0] = parsed[5]
            p2[1, 1] = parsed[6]
            p2[1, 2] = parsed[7]
            p2[1, 3] = parsed[8]
            p2[2, 0] = parsed[9]
            p2[2, 1] = parsed[10]
            p2[2, 2] = parsed[11]
            p2[2, 3] = parsed[12]
            p2[3, 3] = 1
    text_file.close()
    return p2


def project_3d_world(p2, de_center_in_world, w3d, h3d, l3d, ry3d, camera2world):
    """
    help with world
    Projects a 3D box into 2D vertices using the camera2world tranformation
    Note: Since the roadside camera contains pitch and roll angle w.r.t. the ground/world,
    simply adopting KITTI-style projection not works. We first compute the 3D bounding box in ground-coord and then convert back to camera-coord.
    Args:
        p2 (nparray): projection matrix of size 4x3
        de_bottom_center: bottom center XYZ-coord of the object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
        camera2world: camera_to_world translation
    """
    center_world = np.array(de_center_in_world) #bottom center in world
    theta = np.matrix([math.cos(ry3d), 0, -math.sin(ry3d)]).reshape(3, 1)
    theta0 = camera2world[:3, :3] * theta  #first column
    world2camera = np.linalg.inv(camera2world)
    yaw_world_res = math.atan2(theta0[1], theta0[0])
    verts3d = get_camera_3d_8points_g2c(w3d, h3d, l3d,
        yaw_world_res, center_world[:3, :], world2camera, p2, isCenter=False)
    if verts3d is None:
        return None
    verts3d = np.array(verts3d)
    return verts3d


def get_camera_3d_8points_g2c(w3d, h3d, l3d, yaw_ground, center_ground,
                          g2c_trans, p2,
                          isCenter=True):
    """
    function: projection 3D to 2D
    w3d: width of object
    h3d: height of object
    l3d: length of object
    yaw_world: yaw angle in world coordinate
    center_world: the center or the bottom-center of the object in world-coord
    g2c_trans: ground2camera / world2camera transformation
    p2: projection matrix of size 4x3 (camera intrinsics)
    isCenter:
        1: center,
        0: bottom
    """
    ground_r = np.matrix([[math.cos(yaw_ground), -math.sin(yaw_ground), 0],
                         [math.sin(yaw_ground), math.cos(yaw_ground), 0],
                         [0, 0, 1]])
    #l, w, h = obj_size
    w = w3d
    l = l3d
    h = h3d

    if isCenter:
        corners_3d_ground = np.matrix([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                                  [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                                  [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]])
    else:#bottom center, ground: z axis is up
        corners_3d_ground = np.matrix([[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                                  [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                                  [0, 0, 0, 0, h, h, h, h]])

    corners_3d_ground = np.matrix(ground_r) * np.matrix(corners_3d_ground) + np.matrix(center_ground) #[3, 8]

    if g2c_trans.shape[0] == 4: #world2camera transformation
        ones = np.ones(8).reshape(1, 8).tolist()
        corners_3d_cam = g2c_trans * np.matrix(corners_3d_ground.tolist() + ones)
        corners_3d_cam = corners_3d_cam[:3, :]
    else:  #only consider the rotation
        corners_3d_cam = np.matrix(g2c_trans) * corners_3d_ground #[3, 8]

    pt = p2[:3, :3] * corners_3d_cam
    corners_2d = pt / pt[2]
    corners_2d_all = corners_2d.reshape(-1)
    if True in np.isnan(corners_2d_all):
        # print("Invalid projection")
        return None

    corners_2d = corners_2d[0:2].T.tolist()
    for i in range(8):
        corners_2d[i][0] = int(corners_2d[i][0])
        corners_2d[i][1] = int(corners_2d[i][1])
    return corners_2d


def load_detect_data(filename):
    """
    load detection data of kitti format
    """
    data = []
    with open(filename) as infile:
        index = 0
        for line in infile:
            # KITTI detection benchmark data format:
            # (objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry)
            line = line.strip()
            fields = line.split(" ")
            t_data = Data()
            # get fields from table
            t_data.obj_type = fields[
                0].lower()  # object type [car, pedestrian, cyclist, ...]
            t_data.truncation = float(fields[1])  # truncation [0..1]
            t_data.occlusion = int(float(fields[2]))  # occlusion  [0,1,2]
            t_data.obs_angle = float(fields[3])  # observation angle [rad]
            t_data.x1 = int(float(fields[4]))  # left   [px]
            t_data.y1 = int(float(fields[5]))  # top    [px]
            t_data.x2 = int(float(fields[6]))  # right  [px]
            t_data.y2 = int(float(fields[7]))  # bottom [px]
            t_data.h = float(fields[8])  # height [m]
            t_data.w = float(fields[9])  # width  [m]
            t_data.l = float(fields[10])  # length [m]
            t_data.X = float(fields[11])  # X [m]
            t_data.Y = float(fields[12])  # Y [m]
            t_data.Z = float(fields[13])  # Z [m]
            t_data.yaw = float(fields[14])  # yaw angle [rad]
            if len(fields) >= 16:
              t_data.score = float(fields[15])  # detection score
            else:
              t_data.score = 1
            t_data.detect_id = index
            data.append(t_data)
            index = index + 1
    return data


def show_box_with_roll(root_dir, filename, thresh=0.5, projectMethod='World'):
    """show 2d box and 3d box
        yexiaoqing modified
        # 'Ground': using the ground to camera transformation (denorm: the ground plane equation)
        # 'World': using the extrinsics (world to camera transformation)
    """
    image_root = root_dir
    label_dir = os.path.join(image_root,"label")
    extrinsics_dir =  os.path.join(image_root,"extrinsics")
    cal_dir = os.path.join(image_root,"calib")
    thresh = 0.9

    use_denorm = False
    use_extrinsic = False
    if projectMethod == 'Ground':
        use_denorm = True
    elif projectMethod == 'World':
        use_extrinsic = True

    name = filename.split('/')
    name = name[-1].split('.')[0]
    detection_file = os.path.join(label_dir, '%s.txt' % (name))
    result = load_detect_data(detection_file)
    if use_extrinsic:
        extrinsic_file = os.path.join(extrinsics_dir, '%s.yaml' % (name))
        world2camera = read_kitti_ext(extrinsic_file).reshape((4, 4))
        camera2world = np.linalg.inv(world2camera).reshape(4, 4)

    calfile = os.path.join(cal_dir, '%s.txt' % (name))
    p2 = read_kitti_cal(calfile)

    center_pos = []
    for result_index in range(len(result)):
        t = result[result_index]

        # if t.w <= 0.05 and t.l <= 0.05 and t.h <= 0.05:  # invalid annotation
        #     continue

        cam_bottom_center = [t.X, t.Y, t.Z]  # bottom center in Camera coordinate

        if use_extrinsic:
            bottom_center_in_world = camera2world * np.matrix(cam_bottom_center + [1.0]).T
            verts3d = project_3d_world(p2, bottom_center_in_world, t.w, t.h, t.l, t.yaw, camera2world)

        if verts3d is None:
            center_pos.append([])
            continue
        verts3d = verts3d.astype(np.int32)

        center_pos.append((verts3d[0] + verts3d[6]) / 2)
        # print((verts3d[0] + verts3d[6]) / 2, (t.x1 + t.x2) / 2, (t.y1 + t.y2) / 2)

    return center_pos

# show_box_with_roll("1632_fa2sd4a11North151_420_1613710840_1613716786_77_obstacle.txt",projectMethod='World')


def get_calib_from_file(calib_file):
    p2 = []
    with open(calib_file) as f:
        for line in f:
            p2 = line.split(" ")[1:]
            p2 = np.asarray(p2,dtype=float).reshape((3,4))

    return p2


class MyCalibration():
    def __init__(self,root_dir,calib_file):
        p2 = get_calib_from_file(calib_file)
        self.P2 =  p2  # 3 x 4
        self.file_path = calib_file
        self.root_dir = root_dir

    def rect_to_img(self):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        # pts_rect_hom = self.cart_to_hom(pts_rect)
        # pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        # pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        # pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord

        return show_box_with_roll(self.root_dir, self.file_path.split("\\")[-1])

class KITTI(data.Dataset):
    def __init__(self, root_dir, split="sample", cfg={}):
        # basic configuration
        self.num_classes = 9
        self.max_objs = 200
        self.class_name = ["car","van" ,"truck","bus","pedestrian","cyclist","motorcyclist", "barrow", "tricyclist"]
        self.cls2id = {'car': 0, 'van': 1, 'truck': 2,"bus":3,"pedestrian":4,"cyclist":5,"motorcyclist":6,"barrow":7, "tricyclist":8}
        self.resolution = np.array([1280, 384])  # W * H
        self.use_3d_center = cfg['use_3d_center']
        self.writelist = cfg['writelist']
        if cfg['class_merging']:
            self.writelist.extend(['Van', 'Truck'])
        if cfg['use_dontcare']:
            self.writelist.extend(['DontCare'])
        '''    
        ['Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
         'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
         'Cyclist': np.array([1.76282397,0.59706367,1.73698127])] 
        '''
        ##l,w,h
        # self.cls_mean_size = np.array([[1.76255119, 0.66068622, 0.84422524],
        #                                [1.52563191462, 1.62856739989, 3.88311640418],
        #                                [1.73698127, 0.59706367, 1.76282397],
        #                                [1.73698127, 0.59706367, 1.76282397]])
        # self.cls_mean_size = get_mean_size(root_dir,self.class_name)
        self.cls_mean_size = np.asarray([
            [1.27462594, 1.68033459, 4.25404726],
            [1.68234283, 1.74578899, 4.60370771],
            [2.57662322, 2.38113196, 6.0576913],
            [2.95248295, 2.53167071, 10.71633658],
            [1.60362647, 0.48855594, 0.47601192],
            [1.46192315, 0.47453413, 1.50397685],
            [1.41956896, 0.55141885, 1.6601547],
            [1.091107, 0.5064605, 1.000467],
            [1.63675571, 1.09008135, 2.58100218]
        ])

        # data split loading
        # assert split in ['train', 'val', 'trainval', 'test']
        self.split = split

        split_dir = os.path.join(root_dir,  'sample.txt')
        self.idx_list = [x.strip() for x in open(split_dir).readlines()]

        train_ratio = int(len(self.idx_list) * 0.95)
        if self.split == "train":
            self.idx_list = self.idx_list[:train_ratio]
        else:
            self.idx_list = self.idx_list[train_ratio:]

        # path configuration
        self.data_dir = root_dir
        self.image_dir = os.path.join(self.data_dir, 'image')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label')

        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'trainval'] else False
        self.random_flip = cfg['random_flip']
        self.random_crop = cfg['random_crop']
        self.scale = cfg['scale']
        self.shift = cfg['shift']

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # others
        self.downsample = 4

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%s.jpg' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)  # (H, W, 3) RGB mode

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%s.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%s.txt' % idx)
        assert os.path.exists(calib_file)
        return MyCalibration(self.data_dir,calib_file)

    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):

        #  ============================   get inputs   ===========================
        filename = self.idx_list[item]  # index mapping, get real data id
        # image loading
        img = self.get_image(filename)
        img_size = np.array(img.size)

        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size = img_size
        random_crop_flag, random_flip_flag = False, False
        if self.data_augmentation:
            # if np.random.random() < self.random_flip:
            #     random_flip_flag = True
            #     img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.random() < self.random_crop:
                random_crop_flag = True
                crop_size = img_size * np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        coord_range = np.array([center - crop_size / 2, center + crop_size / 2]).astype(np.float32)
        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        calib = self.get_calib(filename)

        features_size = self.resolution // self.downsample  # W * H
        #  ============================   get labels   ==============================
        if self.split != 'test':
            objects = self.get_label(filename)
            # data augmentation for labels
            if random_flip_flag:
                calib.flip(img_size)
                for object in objects:
                    [x1, _, x2, _] = object.box2d
                    object.box2d[0], object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                    object.ry = np.pi - object.ry
                    object.pos[0] *= -1
                    if object.ry > np.pi:  object.ry -= 2 * np.pi
                    if object.ry < -np.pi: object.ry += 2 * np.pi
            # labels encoding
            heatmap = np.zeros((self.num_classes, features_size[1], features_size[0]), dtype=np.float32)  # C * H * W
            size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            offset_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            depth = np.zeros((self.max_objs, 1), dtype=np.float32)
            heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
            heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
            src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
            height2d = np.zeros((self.max_objs, 1), dtype=np.float32)
            cls_ids = np.zeros((self.max_objs), dtype=np.int64)
            indices = np.zeros((self.max_objs), dtype=np.int64)
            mask_2d = np.zeros((self.max_objs), dtype=np.uint8)
            mask_3d = np.zeros((self.max_objs), dtype=np.uint8)
            object_num = len(objects) if len(objects) < self.max_objs else self.max_objs

            # img_center_pos = calib.rect_to_img()
            # if object_num!=len(img_center_pos):
            #     print("*****",object_num,len(img_center_pos))

            for i in range(object_num):
                # filter objects by writelist
                if objects[i].cls_type not in self.writelist:
                    continue

                # filter inappropriate samples by difficulty
                if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                    continue

                # process 2d bbox & get 2d center
                bbox_2d = objects[i].box2d.copy()
                center_3d_x = (bbox_2d[0] + bbox_2d[2]) / 2
                center_3d_y = (bbox_2d[1] + bbox_2d[3]) / 2
                # add affine transformation for 2d boxes.
                bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
                # modify the 2d bbox according to pre-compute downsample ratio
                bbox_2d[:] /= self.downsample

                # process 3d bbox & get 3d center
                center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2],
                                     dtype=np.float32)  # W * H

                # center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
                # center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)


                # center_3d= [img_center_pos[i]]  # project 3D center to image plane
                # center_3d = center_3d[0]  # shape adjustment

                center_3d = [center_3d_x, center_3d_y]
                center_3d = np.asarray(center_3d)

                center_3d = affine_transform(center_3d.reshape(-1), trans)
                center_3d /= self.downsample

                # generate the center of gaussian heatmap [optional: 3d center or 2d center]
                center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)
                if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]: continue
                if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]: continue

                # generate the radius of gaussian heatmap
                w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                radius = gaussian_radius((w, h))
                radius = max(0, int(radius))

                if objects[i].cls_type in ['Van', 'Truck', 'DontCare']:
                    draw_umich_gaussian(heatmap[1], center_heatmap, radius)
                    continue

                cls_id = self.cls2id[objects[i].cls_type]
                cls_ids[i] = cls_id
                draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)

                # encoding 2d/3d offset & 2d size
                indices[i] = center_heatmap[1] * features_size[0] + center_heatmap[0]
                offset_2d[i] = center_2d - center_heatmap
                size_2d[i] = 1. * w, 1. * h

                # encoding depth
                depth[i] = objects[i].pos[-1]

                # encoding heading angle
                heading_angle = objects[i].alpha
                # heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0] + objects[i].box2d[2]) / 2)

                if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
                if heading_angle < -np.pi: heading_angle += 2 * np.pi
                heading_bin[i], heading_res[i] = angle2class(heading_angle)

                # encoding 3d offset & size_3d
                offset_3d[i] = center_3d - center_heatmap
                src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)

                mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
                size_3d[i] = src_size_3d[i] - mean_size

                # objects[i].trucation <=0.5 and objects[i].occlusion<=2 and (objects[i].box2d[3]-objects[i].box2d[1])>=25:
                if objects[i].trucation <= 0.5 and objects[i].occlusion <= 2:
                    mask_2d[i] = 1
            targets = {'depth': depth,
                       'size_2d': size_2d,
                       'heatmap': heatmap,
                       'offset_2d': offset_2d,
                       'indices': indices,
                       'size_3d': size_3d,
                       'offset_3d': offset_3d,
                       'heading_bin': heading_bin,
                       'heading_res': heading_res,
                       'cls_ids': cls_ids,
                       'mask_2d': mask_2d}
        else:
            targets = {}
        # collect return data
        inputs = img
        info = {'img_id': filename,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size / features_size}
        # print(type(inputs),type(calib.P2),type(coord_range))

        return inputs, calib.P2, coord_range, targets, info


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    a = ["car","van" ,"truck","bus","pedestrian","cyclist","motorcyclist", "barrow" , "tricyclist"]
    cfg = {'random_flip': 0.0, 'random_crop': 1.0, 'scale': 0.4, 'shift': 0.1, 'use_dontcare': False,
           'class_merging': False, 'writelist': ["car","van" ,"truck","bus","pedestrian","cyclist","motorcyclist", "barrow" , "tricyclist"], 'use_3d_center': False}
    dataset = KITTI('/home/yang/Desktop/data/3D/data/', 'sample', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    for batch_idx, (inputs, P2, coord_range, targets, info) in enumerate(dataloader):
        print(type(P2))
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.uint8))

        print(type(targets['size_3d'][0][0]))

        # test heatmap
        heatmap = targets['heatmap'][0]  # image id
        heatmap = Image.fromarray(heatmap[0].numpy() * 255)  # cats id
        # img.show()
        # heatmap.show()
        print(info["img_id"])


        break

    # print ground truth fisrt
    # objects = dataset.get_label(info["img_id"][0])
    # for object in objects:
    #     print(object.to_kitti_format())
