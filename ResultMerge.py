"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import sys
sys.path.append('./markrcnn_benchmark/DOTA_devkit')
import os
import numpy as np
import pandas as pd
import DOTA_devkit.dota_utils as util
import re
import time
from DOTA_devkit import polyiou
from math import pi
import cv2
from shapely.geometry import Polygon
from shapely.affinity import rotate

## the thresh for nms when merge image
nms_thresh = 0.3

def py_cpu_nms_poly(dets, thresh):
    scores = dets[:, 8]
    polys = []
    areas = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        for j in range(order.size - 1):
            iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
            ovr.append(iou)
        ovr = np.array(ovr)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ## index for dets
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def nmsbynamedict(nameboxdict, nms, thresh):
    nameboxnmsdict = {x: [] for x in nameboxdict}
    for imgname in nameboxdict:
        #print('imgname:', imgname)
        #keep = py_cpu_nms(np.array(nameboxdict[imgname]), thresh)
        #print('type nameboxdict:', type(nameboxnmsdict))
        #print('type imgname:', type(imgname))
        #print('type nms:', type(nms))
        keep = nms(np.array(nameboxdict[imgname]), thresh)
        #print('keep:', keep)
        outdets = []
        #print('nameboxdict[imgname]: ', nameboxnmsdict[imgname])
        for index in keep:
            # print('index:', index)
            outdets.append(nameboxdict[imgname][index])
        nameboxnmsdict[imgname] = outdets
    return nameboxnmsdict
def poly2origpoly(poly, x, y, rate):
    origpoly = []
    for i in range(int(len(poly)/2)):
        tmp_x = float(poly[i * 2] + x) / float(rate)
        tmp_y = float(poly[i * 2 + 1] + y) / float(rate)
        origpoly.append(tmp_x)
        origpoly.append(tmp_y)
    return origpoly

def mergebase(srcpath, dstpath, nms):
    filelist = util.GetFileFromThisRootDir(srcpath)
    print(filelist)
    # PLUS (0314)
    filelist = [file for file in filelist if file.split('.')[-1] == 'txt']
    for fullname in filelist:
        name = util.custombasename(fullname)
        #print('name:', name)
        dstname = os.path.join(dstpath, name + '.txt')
        print(dstname)
        with open(fullname, 'r', encoding='utf-8') as f_in:
            nameboxdict = {}
            lines = f_in.readlines()
            splitlines = [x.strip().split(' ') for x in lines]
            for splitline in splitlines:
                subname = splitline[0]
                # print(subname)
                splitname = subname.split('__')
                oriname = splitname[0]
                pattern1 = re.compile(r'__\d+___\d+')
                #print('subname:', subname)
                x_y = re.findall(pattern1, subname)
                x_y_2 = re.findall(r'\d+', x_y[0])
                x, y = int(x_y_2[0]), int(x_y_2[1])

                pattern2 = re.compile(r'__([\d+\.]+)__\d+___')

                rate = re.findall(pattern2, subname)[0]

                confidence = splitline[1]
                poly = list(map(float, splitline[2:]))
                origpoly = poly2origpoly(poly, x, y, rate)
                det = origpoly
                det.append(confidence)
                det = list(map(float, det))
                if (oriname not in nameboxdict):
                    nameboxdict[oriname] = []
                nameboxdict[oriname].append(det)
            nameboxnmsdict = nmsbynamedict(nameboxdict, nms, nms_thresh)
            with open(dstname, 'w') as f_out:
                for imgname in nameboxnmsdict:
                    for det in nameboxnmsdict[imgname]:
                        #print('det:', det)
                        confidence = det[-1]
                        bbox = det[0:-1]
                        outline = imgname + ' ' + str(confidence) + ' ' + ' '.join(map(str, bbox))
                        #print('outline:', outline)
                        f_out.write(outline + '\n')
def mergebyrec(srcpath, dstpath):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    mergebase(srcpath,
              dstpath,
              py_cpu_nms)
def mergebypoly(srcpath, dstpath):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    mergebase(srcpath,
              dstpath,
              py_cpu_nms_poly)

def poly (rbox):
   # rbox : [x1, y1, x2, y2, x3, y3, x4, y4]
   return Polygon([(rbox[0], rbox[1]), (rbox[2], rbox[3]), (rbox[4], rbox[5]), (rbox[6], rbox[7])])


def half_iou(rbox1, rbox2):
   poly1 = poly(rbox1)
   poly2 = poly(rbox2)
   
   if poly1.area >= poly2.area:
       return poly1.intersection(poly2).area / poly2.area
   else :
       return poly2.intersection(poly1).area / poly1.area

if __name__ == '__main__':
    before_path = 'exp_dacon/dacon/inference/dacon_test_cut'
    after_path = 'exp_dacon/dacon/inference/dacon_test_cut/results'
    mergebypoly(before_path, after_path)

    result_path = 'exp_dacon/dacon/inference/dacon_test_cut/results'
    txts = os.listdir(result_path)

    cls = {
        "container":1,
        "oil-tanker":2,
        "aircraft-carrier":3,
        "maritime-vessels":4
    }

    i = 0
    for txt in txts:
        txt_path = os.path.join(result_path,txt)
        txt_name = txt.strip('.txt')
        df = pd.read_csv(txt_path,header=None,sep=' ')
        df.columns = ['file_name','confidence','point1_x','point1_y','point2_x','point2_y',
                      'point3_x','point3_y','point4_x','point4_y']
        df['file_name'] = df['file_name'].astype(str) + '.png'
        class_id = cls[txt_name]
        df['class_id'] = class_id
        if i == 0:
            test_df = df.copy()
            i += 1
        else:
            test_df = pd.concat([test_df,df],axis=0)
    test_df = test_df[['file_name','class_id','confidence','point1_x','point1_y',
                       'point2_x','point2_y','point3_x','point3_y','point4_x','point4_y']]
    test_df = test_df.sort_values(['file_name','class_id','confidence'])
    test_df.reset_index(drop=True,inplace=True)
    test_df.to_csv('submission_noiou.csv',index=False)
    

    df = test_df.copy()
    df['NMS'] = True
    halfiou_thresh = {1: 0.7, 2: 0.7, 3: 0.1, 4: 0.9}
    new_df = pd.DataFrame(columns = ['file_name', 'class_id','confidence', 'point1_x', 'point1_y', 'point2_x', 'point2_y', 
                                       'point3_x', 'point3_y', 'point4_x', 'point4_y', 'NMS'])

    for cls in df.class_id.unique():
        df_cls = df[df.class_id == cls]
        image_list = df_cls.file_name.unique()

        for img in image_list:
            idx_list = list(df_cls[df_cls.file_name == img].index)
            for idx in idx_list:
                rbox1 = [df_cls['point1_x'][idx], df_cls['point1_y'][idx], df_cls['point2_x'][idx], df_cls['point2_y'][idx], df_cls['point3_x'][idx], df_cls['point3_y'][idx], df_cls['point4_x'][idx], df_cls['point4_y'][idx]]
                score1 = df_cls['confidence'][idx]
                for idx2 in idx_list[idx_list.index(idx)+1:]:
                    rbox2 = [df_cls['point1_x'][idx2], df_cls['point1_y'][idx2], df_cls['point2_x'][idx2], df_cls['point2_y'][idx2], df_cls['point3_x'][idx2], df_cls['point3_y'][idx2], df_cls['point4_x'][idx2], df_cls['point4_y'][idx2]]
                    score2 = df_cls['confidence'][idx2]

                    if half_iou(rbox1, rbox2) >= halfiou_thresh[cls] :
                        if score1 == score2:
                            if poly(rbox1).area >= poly(rbox2).area:
                                df_cls['NMS'][idx2] = False
                            else:
                                df_cls['NMS'][idx] = False
                        elif score1 > score2:
                            df_cls['NMS'][idx2] = False
                        else :
                            df_cls['NMS'][idx] = False

        new_df = pd.concat([new_df, df_cls]).reset_index(drop=True)
    new_df = new_df[new_df.NMS == True].reset_index(drop=True)
    new_df.drop('NMS', axis=1, inplace=True)
    new_df.to_csv('submission.csv',index=False)

