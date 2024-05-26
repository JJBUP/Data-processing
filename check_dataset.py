import functools
import numpy as np
import pathlib
from tqdm import tqdm
from multiprocessing import Pool

"""
检查数据是否包含多余列，标签是否错误，并保存到新的save_root去除多余文件
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
def data_process(paths, save_root):
    # all data unit
    for path in tqdm(paths):
        # one data unit
        for txt_path in path.rglob("Annotations/*.txt"):
            # 检查标签错误
            try:
                assert txt_path.stem in names, "文件名错误：" + str(txt_path)
            except Exception as e:
                print(e)
                log_path = pathlib.Path("./log/error_log.txt")
                with open(log_path, 'w') as f:
                    f.write(str(txt_path) + "\n")
            part_data = np.loadtxt(txt_path)[:, :6]  # 去除多余列
            part_data = part_data
            save_path = pathlib.Path(str(txt_path).replace(data_root, save_root))
            # 保存
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True)
            np.savetxt(save_path, part_data, fmt='%.8f %.8f %.8f %d %d %d')


# 多进程
def multi_data_process(path, save_root):
    # one data unit
    for txt_path in path.rglob("Annotations/*.txt"):
        # 检查标签错误
        try:
            assert txt_path.stem in names, "文件名错误：" + str(txt_path)
        except Exception as e:
            print(e)
            log_path = pathlib.Path("./error_log.txt")
            with open(log_path, 'w') as f:
                f.write(str(txt_path) + "\n")
        if save_root is not None or not "":
            part_data = np.loadtxt(txt_path)[:, :6]  # 去除多余列
            part_data = part_data
            save_path = pathlib.Path(str(txt_path).replace(data_root, save_root))
            # 保存
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True)
            np.savetxt(save_path, part_data, fmt='%.8f %.8f %.8f %d %d %d')
    return


if __name__ == "__main__":
    data_root = "/data/Zhong_Shui/Pro_Data/"  # 数据集root
    save_root = "/data/Zhong_Shui/Pro_Data_preprocess/"  # 检查后数据集保存root，设置为None或''则仅检查不保存
    # save_root = "" # 检查后数据集保存root，设置为None或''则仅检查不保存
    root_path = pathlib.Path(data_root)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    paths = []
    # mini data unit
    print("processing...")

    for work in root_path.glob('Work_*'):
        for area in work.glob("Area_*"):
            area_child = list(p for p in area.glob('Area_*_*'))
            if area_child:  # /Area_*/Area_*_*/
                paths.extend(area_child)
            else:
                paths.append(area)  # Area_*/
    pool = Pool(processes=32)
    partial_data_process = functools.partial(multi_data_process, save_root=save_root)
    result = list(tqdm(pool.imap(partial_data_process, paths), total=len(paths)))
    pool.close()
    pool.join()
    # data_process(paths, save_root)
    print("done!")
