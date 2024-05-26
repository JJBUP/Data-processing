"""
Author:Bitor
According proportion split train and test of dataset.
"""
from math import sqrt
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pathlib
import functools
import torch
from tqdm import tqdm

# 数据集的类别名称
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


def get_loss(s, val_propotion):  # 计算所有目标平方和损失的平均，保证接近val_propotion
    return sqrt(np.nanmean((s- val_propotion) ** 2))


def partition_bellman(df, val_propotion):
    marked = np.zeros(df.shape[0], dtype=bool)
    best_sum = np.zeros(df.shape[-1])
    best_loss = get_loss(best_sum, val_propotion)
    has_better = True  # 是否有更好的解
    df.loc[:, :] = df.values / df.values.sum(0)
    while has_better:
        has_better = False
        for i in range(df.shape[0]):
            if marked[i]:
                # 如果当前item已经被选中，是否可以通过踢出该item来减小loss
                sum = best_sum - df.iloc[i].values
                loss = get_loss(sum, val_propotion)
                if loss < best_loss:
                    marked[i] = False
                    best_sum, best_loss = sum, loss
                    has_better = True  # 这一轮迭代出现了更好的loss解
            else:
                # 如果当前item没有被选中，是否可以通过选择该item来减小loss
                sum = best_sum + df.iloc[i].values
                loss = get_loss(sum, val_propotion)
                if loss < best_loss:
                    marked[i] = True
                    best_sum, best_loss = sum, loss
                    has_better = True  # 这一轮迭代出现了更好的loss解
    unmarked = np.logical_not(marked)
    # 收集所有work
    # show current work
    selected_path = df.iloc[marked]
    unselected_path = df.iloc[unmarked]
    selected_sum = selected_path.values.sum(0)  # 每个类别的比例
    selected_loss = np.sqrt(((selected_sum - val_propotion) ** 2))
    prop = np.nanmean(selected_sum)
    loss = get_loss(selected_sum, val_propotion)

    selected_sum = pd.Series(selected_sum, index=df.columns)
    selected_loss = pd.Series(selected_loss, index=df.columns)
    print(f"proportion = {prop}",
          selected_sum,
          f"loss = {loss}",
          selected_loss,
          selected_path.index,
          unselected_path.index,
          sep='\n')
    selected_paths = list(df.iloc[marked].index.values)
    unselected_paths = list(df.iloc[unmarked].index.values)
    return selected_paths, unselected_paths,selected_path,unselected_path


def multi_partition_bellman(df_dict, val_propotion):
    selected_paths_all_work = []
    unselected_paths_all_work = []
    sel_data,unsel_data = [],[]
    for df_key in df_dict:  # 依次处理每个work
        df = df_dict[df_key]
        print(f">>>>>>>{str(df_key)}>>>>>>>")
        selected_paths, unselected_paths,selected_path,unselected_path = partition_bellman(df, val_propotion)
        sel_data.append(selected_path)
        unsel_data.append(unselected_path)
    selected_paths_all_work += selected_paths
    unselected_paths_all_work += unselected_paths
    a = pd.concat(sel_data, axis=0).sum(0)
    b = pd.concat(sel_data+unsel_data, axis=0).sum(0)
    print(a/b)
    return selected_paths_all_work, unselected_paths_all_work


def save_dataset(path, save_root, scale=1, suffix=".npy"):
    if not isinstance(save_root, pathlib.Path):
        save_root = pathlib.Path(save_root)
    assert save_root.exists(), "path not exists{save_root}"
    name2id = {name: id for id, name in enumerate(names)}
    # for path in tqdm(paths):
    # 处理每个小场景
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)
    txt_paths = path.rglob("*.txt")
    area_points = []
    for txt_path in txt_paths:
        points = np.loadtxt(txt_path)
        # 获取类别名称
        assert txt_path.stem in names, "未知类别：{}！".format(txt_path)
        labels = np.ones((points.shape[0], 1)) * name2id[txt_path.stem]
        area_points.append(np.concatenate((points, labels), axis=1))
    area_points = np.concatenate(area_points, axis=0)
    area_points[:, :3] = (area_points[:, :3] - area_points[:, :3].min(axis=0)) * scale  # 平移+缩放
    area_points = area_points.astype(np.float32)
    # 保存
    # 名称+替换路径
    # path = /data/Zhong_Shui/Pro_Data/Work_1/Area_8/Area_8_16' to /data/Zhong_Shui/trainval/train/Work1_Area_8_16.npy'
    start_idx = str(path).find("Work")
    save_path = save_root / (str.join("_", [str(path)[start_idx:start_idx + 6], path.stem]) + suffix)
    if ".txt" == suffix:
        np.savetxt(save_path, area_points, fmt='%.8f %.8f %.8f %d %d %d %d')
    elif ".npy" == suffix:
        np.save(save_path, area_points)
    elif ".pth" == suffix:
        # torch.save(torch.from_numpy(area_points).type(torch.float32),save_path)
        torch.save(area_points, save_path)
    else:
        raise ValueError("无效的后缀", suffix)


if __name__ == '__main__':
    val_propotion = 0.25  # 验证集比例
    scale = 0.1  # 缩放规模
    csv_root = pathlib.Path("./")  # collect_split_dataset.py的csv文件
    train_root = pathlib.Path("/data/Zhong_Shui/zhongshui_pth/train")  # train保存路径
    val_root = pathlib.Path("/data/Zhong_Shui/zhongshui_pth/val")  # val保存路径
    suffix = ".pth"  # suffix 文件后缀.txt,.pth或.npy
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # csv_paths = csv_root.rglob("Work_4.csv")
    csv_paths = csv_root.rglob("*point_num_ratio.csv")
    df_dict = {k: pd.read_csv(k, index_col=0) for k in csv_paths}
    if not train_root.exists():
        train_root.mkdir(parents=True)
    if not val_root.exists():
        val_root.mkdir(parents=True)
    selected_paths, unselected_paths = multi_partition_bellman(df_dict, val_propotion)
    pool = Pool(processes=32)
    partial_save_train = functools.partial(save_dataset, save_root=train_root, scale=1, suffix=suffix)
    partial_save_val = functools.partial(save_dataset, save_root=val_root, scale=1, suffix=suffix)
    # list(tqdm(pool.imap(partial_save_train, unselected_paths), total=len(unselected_paths)))
    # list(tqdm(pool.imap(partial_save_val, selected_paths), total=len(selected_paths)))
    pool.close()
    pool.join()
