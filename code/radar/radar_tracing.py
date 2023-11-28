# -*- coding: utf-8 -*-
# @Time    : 2023/9/26 12:18
# @Name    : radar_tracing.py
# @email   : yangemail2@163.com
# @Author  : haoyunjixiang

import os
import cv2
import numpy as np


def frame_to_img(coor_list,id_list):
    x_list = (coor_list[:,1] - min(coor_list[:,1])) * 10
    y_list = coor_list[:,0] * 10

    max_x = int(max(x_list))
    max_y = int(max(y_list))
    print(max_x,max_y,x_list[0],y_list[0])
    print(max_x, max_y, x_list[1], y_list[1])
    img = np.zeros((max_x+1,max_y+1,3))

    colors = [np.random.randint(160,255,3) for i in range(len(id_list))]
    for id,x,y in zip([j for j in range(len(id_list))],x_list,y_list):
        color = colors[id]
        # print(id,int(id_list[id]),x,y,color)
        img[int(x),int(y)] = color
    # cv2.imshow("t",img)
    # cv2.waitKey(0)
    cv2.imwrite("t.jpg",img)


def read_csv():
    fiel_path = "E:\Learn\data\\3D\数据集3 初赛A榜\\RadarData.csv"
    csv_file = open(fiel_path)
    data_dict = {}
    for line in csv_file:
        if line.__contains__("timestamp"):
            continue
        if line.__contains__("1688526705"):
            break

        line_list = line.strip().split(",")

        # if abs(float(line_list[4])) < 0.5:
        #     continue

        if not data_dict.keys().__contains__(line_list[0]):
            data_dict[line_list[0]] = [[int(line_list[1]), float(line_list[2]), float(line_list[3]), float(line_list[4]), float(line_list[5])]]
        else:
            data_dict[line_list[0]].append(
                [int(line_list[1]), float(line_list[2]), float(line_list[3]), float(line_list[4]), float(line_list[5])])

    # print(data_dict["1688526704"])
    one_time_data = np.asarray(data_dict["1688526704"])
    print(one_time_data.shape)
    coor_list = one_time_data[:,1:3]
    id_list = one_time_data[:,0]

    for i in range(len(id_list)):
        print(data_dict["1688526704"][i])

    frame_to_img(coor_list,id_list)

# read_csv()

# fiel_path = "E:\Learn\data\\3D\数据集3 初赛A榜\\RadarData.csv"
# csv_file = open(fiel_path)
# data_dict = {}
# count = 0
# for line in csv_file:
#     count = count + 1
# print(count)

mystr = "1111111111.png"
print(mystr.replace(".png",".jpg"))



