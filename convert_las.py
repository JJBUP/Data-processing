import functools
import numpy as np
import pathlib
from tqdm import tqdm
from multiprocessing import Pool
import laspy
"""
将单区域数据合并，并保存为las，加入额外地理信息
"""
names = [
    'Shed',
    'Concretehouse',  # 居民地
    'Cementroad',
    'Dirtroad',  # 交通
    'Reinforcedslope',
    'Reinforcedscarp',
    'Dam',  # 地貌
    'Vegetablefield',
    'Grassland',
    'Dryland',
    'Woodland',
    'Bareland',  # 植被与土质
    'Waterline',
    'Ditch',  # 水系
    'Others'  # 其他
]

# 单进程
def data_process(paths:(pathlib.Path), save_root,header_dict):
    # all data unit
    for path in tqdm(paths):
        # one data unit
        complete_data = []
        for txt_path in path.rglob("Annotations/*.txt"):
            # 检查标签错误
            try:
                assert txt_path.stem in names, "文件名错误：" + str(txt_path)
            except Exception as e:
                print(e)
                log_path = pathlib.Path("./error_log.txt")
                with open(log_path, 'w') as f:
                    f.write(str(txt_path) + "\n")
            part_data = np.loadtxt(txt_path)[:, :6]  # 去除多余列
            complete_data.append(part_data)
        paths_parts = list(path.parts)
        # 配对header
        try:
            for s in paths_parts:
                if "Work" in s:
                    header = header_dict[s]
        except Exception as e:
            print("不存在带有work的名字")
        complete_data = np.concatenate(complete_data,axis=0)
        las = convert_las(complete_data,header)
        # 路径
        len_root = len(list(pathlib.Path(data_root).parts))
        paths_parts = list(pathlib.Path(save_root).parts) + paths_parts[len_root:]
        _name = paths_parts[len_root:]
        _name.append(".las")
        paths_parts.append(str.join("_",_name))
        save_path = pathlib.Path(*paths_parts)      
        # 保存
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        las.write(save_path) #保存las文件

# 多进程
def multi_data_process(path:pathlib.Path, save_root,header_dict):
    # one data unit
    complete_data = []
    for txt_path in path.rglob("Annotations/*.txt"):
        # 检查标签错误
        try:
            assert txt_path.stem in names, "文件名错误：" + str(txt_path)
        except Exception as e:
            print(e)
            log_path = pathlib.Path("./error_log.txt")
            with open(log_path, 'w') as f:
                f.write(str(txt_path) + "\n")
        part_data = np.loadtxt(txt_path)[:, :6]  # 去除多余列
        complete_data.append(part_data)
    paths_parts = list(path.parts)
    # 配对header
    try:
        for s in paths_parts:
            if "Work" in s:
                header = header_dict[s]
    except Exception as e:
        print("不存在带有work的名字")
    complete_data = np.concatenate(complete_data,axis=0)
    las = convert_las(complete_data,header)
    # 路径
    len_root = len(list(pathlib.Path(data_root).parts))
    paths_parts = list(pathlib.Path(save_root).parts) + paths_parts[len_root:]
    _name = paths_parts[len_root:]
    _name.append(".las")
    paths_parts.append(str.join("_",_name))
    save_path = pathlib.Path(*paths_parts)            
    # 保存
    if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
    las.write(save_path) #保存las文件


#将预测结果保存为las文件
def convert_las(points, header):
    
    # 生成las 设置las头，设置数据值
    las = laspy.LasData(header)
    las.x = points[:, 0]  # xyz [N,3]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.red = points[:, 3] * 255  # rgb 将颜色值 x 255 las需要
    las.green = points[:, 4] * 255
    las.blue = points[:, 5] * 255
    return las

if __name__ == "__main__":
    data_root = "/data/Zhong_Shui/WCS3D_v2_fix_txt"  # 数据集root
    save_root = "/data/Zhong_Shui/WCS3D_v2_fix_las"  # 检查后数据集保存root，设置为None或''则仅检查不保存
    work1_las = "/data/Zhong_Shui/WCS3D_ori_las/20210318三维点云大藤峡水利枢纽工程重点施工区无人机航摄第九十四期.las"
    work2_las = "/data/Zhong_Shui/WCS3D_ori_las/东干渠-珠基高.las"
    work3_las = "/data/Zhong_Shui/WCS3D_ori_las/220716_PSLW_20cm.las"
    # save_root = "" # 检查后数据集保存root，设置为None或''则仅检查不保存
    root_path = pathlib.Path(data_root)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    paths = []
    # mini data unit
    print("processing...")

    work1_las_header = laspy.LasHeader(version="1.4", point_format=2)
    work2_las_header = laspy.LasHeader(version="1.4", point_format=2)
    work3_las_header = laspy.LasHeader(version="1.4", point_format=2)
    work1_las_header.vlrs = laspy.open(work1_las).header.vlrs
    work2_las_header.vlrs = laspy.open(work2_las).header.vlrs
    work3_las_header.vlrs = laspy.open(work3_las).header.vlrs
    header_dict = {
        "Work_1": work1_las_header,
        "Work_2": work2_las_header,
        "Work_3": work3_las_header,
    }

    for work in root_path.glob('Work_*'):
        for area in work.glob("Area_*"):
            area_child = list(p for p in area.glob('Area_*_*'))
            if area_child:  # /Area_*/Area_*_*/
                paths.extend(area_child)
            else:
                paths.append(area)  # Area_*/
    pool = Pool(processes=32)
    partial_data_process = functools.partial(multi_data_process, save_root=save_root,header_dict=header_dict)
    result = list(tqdm(pool.imap(partial_data_process, paths), total=len(paths)))
    pool.close()
    pool.join()
    # data_process(paths, save_root,header_dict)
    print("done!")
