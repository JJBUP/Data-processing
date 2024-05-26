"""
Author:Bitor
统计数据，保存csv文件，划分train/val
"""
import functools
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from partition_dataset_bellman import partition_bellman,save_dataset
from multiprocessing import Pool

# 类别名
names = [
    'Shed',
    'Concretehouse',  # 居民地
    'Cementroad',
    'Dirtroad',  # 交通
    'Reinforcedslope',
    'Reinforcedscarp',
    'Dam',  # 地貌
    # 'Electrictower',  # 管线
    'Vegetablefield',
    'Grassland',
    'Dryland',
    'Woodland',
    'Bareland',  # 植被与土质
    'Waterline',
    'Ditch',  # 水系
    'Others'  # 其他
]
def collect_dataset(work_path):
    # df_dict = {}
    # for work in root_path.rglob('Work_*'):
        paths = []
        for area in work_path.iterdir():
            area_child = list(p for p in area.rglob('Area_*_*') if p.is_dir())
            if area_child:
                paths.extend(area_child)
            else:
                paths.append(area)

        df = pd.DataFrame(0, index=paths, columns=names, dtype=int)
        for path in tqdm(paths):
            for label_path in path.rglob('*.txt'):
               label = label_path.stem
               if label in names:
                    df.loc[path, label] = sum(1 for _ in label_path.open())   
        # 保存
        df.to_csv(f'{work_path.stem}_point_num.csv')
        (df / df.sum(0)).to_csv(f'{work_path.stem}_point_num_ratio.csv')
        return work_path,(df / df.sum(0))
    #     df_dict[f"{work.stem}"] = df
    # return df_dict


if __name__ == '__main__':
    val_propotion = 0.25 # 验证集比例
    scale = 1 # 缩放规模
    train_root = "/data/Zhong_Shui/trainval_0.25_pth/train" # train保存路径
    val_root = "/data/Zhong_Shui/trainval_0.25_pth/val" # val保存路径
    root_path = Path('/data/Zhong_Shui/Pro_Data_preprocess').expanduser()  # 修改为数据路径
    suffix = ".pth"# suffix 文件后缀.txt/.npy/.pt
    # 统计数据
    work_paths = [work for work in root_path.rglob('Work_*')]
    pool = Pool(processes=len(work_paths))
    df_dict ={ k:v for k,v in (pool.imap(collect_dataset, work_paths))}
    pool.close()
    pool.join()
    print("collect done!")
    # df_dict = collect_dataset(root_path)
    # 划分
    pool = Pool(processes=len(work_paths))
    selected_paths,unselected_paths = partition_bellman(df_dict,val_propotion)
    pool = Pool(processes=32)
    if not os.path.exists(train_root):
        os.makedirs(train_root)
    if not os.path.exists(val_root):
        os.makedirs(val_root)
    partial_save_train = functools.partial(save_dataset,save_root = train_root ,scale = 1,suffix=suffix)
    partial_save_val = functools.partial(save_dataset,save_root = val_root ,scale = 1,suffix=suffix)
    list(tqdm(pool.imap(partial_save_train, unselected_paths), total=len(unselected_paths)))
    list(tqdm(pool.imap(partial_save_val, selected_paths), total=len(selected_paths)))
    pool.close()
    pool.join()