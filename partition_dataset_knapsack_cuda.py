"""
Author:Jjbinary
According proportion split train and test of dataset.
"""
import numpy as np
import torch as tc
import pandas as pd
import pathlib
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

def get_weight(data):
    """
    在展示selected选中部分的时候，通过累计每个类别的权重，查看权重分布是否均匀合理
    """
    return np.nansum(data, axis=0)

def get_weight_tc(data):
    """
    在展示selected选中部分的时候，通过累计每个类别的权重，查看权重分布是否均匀合理
    """
    return tc.nansum(data, dim=0)

def get_loss(data, val_proportion, mode=""):
    """
    mode:两种模式
        ""：默认模式，计算dp与指定比例的差距损失
        "acc": 在展示selected选中部分的时候，通过累计每个类别的loss，查看loss分布是否均匀合理
    """
    # 损失计算，即累计权重与指定val的差距
    if mode == "acc":
        return np.sqrt(np.power(np.nansum(data, axis=0) - val_proportion, 2))  # 计算select数据loss
    return np.sqrt(np.nanmean(np.power(data - val_proportion, 2), axis=-1))  # 计算dp loss

def get_loss_tc(data, val_proportion, mode=""):
    """
    mode:两种模式
        ""：默认模式，计算dp与指定比例的差距损失
        "acc": 在展示selected选中部分的时候，通过累计每个类别的loss，查看loss分布是否均匀合理
    """
    # 损失计算，即累计权重与指定val的差距
    if mode == "acc":
        return tc.sqrt(tc.pow(tc.nansum(data, dim=0) - val_proportion, 2))  # 计算select数据loss
    return tc.sqrt(tc.nansum(tc.pow(data - val_proportion, 2), dim=-1)/(~tc.isnan(data)).sum(-1))  # 计算dp loss,由于tc2.1才有nan，所以采用组合的方式

def get_var(data):
    # 多目标数据方差
    E = tc.nansum(data, dim=-1)/(~tc.isnan(data)).sum(-1)
    Var = tc.nansum(tc.pow(data - E.reshape(E.shape[0],E.shape[1],1), 2), dim=-1)/(~tc.isnan(data)).sum(-1)
    return tc.sqrt(Var)

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

def get_best_select(data, val_proportion, stride, use_var, loss_tolerance):
    # 动态规划解决数据集划分
    data = tc.from_numpy(data).cuda()
    # 优化计算，val超过0.5则将train与val比例置换
    selection = {}
    change = False
    if val_proportion > 0.5:
        val_proportion = 1 - val_proportion
        change = True

    n, m = data.shape[0], data.shape[1]
    # 权重格式化
    weight_cls = tc.zeros((n + 1, m)).cuda()
    weight_cls[1:] = data

    # 创建一个二维数组来存储子问题的解 数量*依次递减的loss
    prop_quant = len([_ for _ in tc.arange(0, val_proportion, stride)])
    dp_w = tc.zeros((n + 1, prop_quant + 1, m)).cuda()  # 比例为主要状态,m细致的存储了每一个cls的权重
    dp_seq = tc.full((n + 1, prop_quant + 1, n), False, dtype=bool).cuda()
    # 填充数组，从底部开始
    # loss渐进式，保证每个物体都能选择最合适的值来靠近val，但是存在缺陷
    # 由于dp_w保存的是每一个目标的权重，动态的get_loss获得值的并不稳定
    # 因为权重是动态累计的，那个位置的权重与现在的权重累计的loss更小是说不准的
    #      val_prop  --->
    # o   0.0 0.01 0.02 0.03 ....
    # b 0  w    w    w    w
    # j 1  w    w    w    w
    # | 2  w    w    w    w
    # | 3  w    w    w    w
    # v 4  w    w    w    w

    for i in tqdm(range(1, n + 1)):  # 从编号为1取数据,保证dp数组中的存在第0个数据以便加减获取
        for p in range(prop_quant + 1):
            prop = p * stride  # 当前loss
            # 注意，选择最小loss的时候只能选择左边
            # 优化：为了削弱顺序性，我们从前n-1个数据中选择最小值

            # 使用：相加或替换
            former = get_loss_tc(dp_w[i - 1][p], prop)  # [i-1][l]肯定是目前该位置loss的最小，但是动态的loss获取不一定是由它产生的
            # 相加的最小值
            # min_idx = np.argmin(get_loss(weight_cls[i] + dp_w[:i], loss_pro))
            # 替换的最小值
            # i种替换情况+1中相加情况 , 前i个物体种寻找做小loss, loss划分比例, 具体选了那些
            r, c = 0,0
            min_loss = 1
            w_select,seq_select = torch.tensor([]).cuda(),torch.tensor([]).cuda()
            for j in range(0, i+1):  # 当前物体编号i-1，之前的物体编号从0到i-2
                dp_seq_select = dp_seq[:i].clone() # torch内存公用
                if j < i:
                    dp_seq_select[:, :, j] = False # 循环设置每一行为False(删除该数据),最后保留一个完整的不设置的
                rep = (dp_seq_select.shape[0], dp_seq_select.shape[1], 1, 1)
                # 选择权重
                dp_w_select = tc.tile(weight_cls[1:][None, None, :], rep)
                dp_w_select[~dp_seq_select] = 0  # 将False的地方设置为0
                dp_w_select = dp_w_select.sum(-2) # 累计权重
                dp_loss_selelct = get_loss_tc(weight_cls[i] + dp_w_select, prop)
                min_index = dp_loss_selelct.argmin()
                _r, _c = unravel_index(min_index, dp_loss_selelct.shape)
                if dp_loss_selelct[_r, _c] < min_loss:
                    r, c = _r, _c
                    min_loss = dp_loss_selelct[r, c]
                    w_select = dp_w_select[r, c]
                    seq_select = dp_seq_select[r, c]
            latter = min_loss
            if former < latter:
                dp_w[i][p] = dp_w[i - 1][p]
                dp_seq[i][p] = dp_seq[i - 1][p]
            else:
                dp_w[i][p] = weight_cls[i] + w_select
                dp_seq[i][p] = seq_select
                dp_seq[i][p][i - 1] = True  # 物体序号从0开始
    dp_loss = get_loss_tc(dp_w, val_proportion)
    dp_var = get_var(dp_w)  # 求得平均方差
    # 选择 dp_loss 和 dp_var 方差都尽可能小的方案
    if use_var:  # 是否考虑方差
        if loss_tolerance is not None:  # 是否现选出一部分小的loss，再同时优化损失和方差
            if loss_tolerance >= 1:
                # 设定loss最小值为容忍阈值
                condidate_mask = dp_loss < dp_loss.min() * loss_tolerance  # 优先考虑loss尽可能小
            elif loss_tolerance < 1:
                # 手动设置阈值
                condidate_mask = dp_loss < loss_tolerance
            else:
                raise ("loss_tolerance format is not correct !")
            if condidate_mask.sum() == 0:
                raise ("loss_tolerance section don't have data")
            # 不同源目标统一量纲后优化
            loss_normal = dp_loss[condidate_mask] / dp_loss[condidate_mask].sum()
            vars_normal = dp_var[condidate_mask] / dp_var[condidate_mask].sum()
            index = (loss_normal + vars_normal).argmin()
            # 在原始数据中找到对应的索引
            ori_indices_r, ori_indeces_c = tc.where(condidate_mask)  # 原始行索引和列索引
            r, c = ori_indices_r[index], ori_indeces_c[index]
            # assert dp_loss[condidate_mask][index] == dp_loss[r][c] # 恢复成功了
        else:
            # loss 与 var 一视同仁
            loss_normal = dp_loss / dp_loss.sum()
            vars_normal = dp_var / dp_var.sum()
            index = (loss_normal + vars_normal).argmin()
            r, c = index // (prop_quant + 1), index % (prop_quant + 1)
    else:  # 不考虑方差
        index = dp_loss.argmin()
        r, c = index // (prop_quant + 1), index % (prop_quant + 1)

    # 考虑或不考虑方差（使用）
    prop_best = tc.nansum(dp_w[r][c])/(~tc.isnan(dp_w[r][c])).sum()  # 考虑方差和损失的最佳比例
    loss_best = dp_loss[r][c]  # 考虑方差和损失的最佳损失
    var_best = dp_var[r][c]  # 考虑方差和损失的最佳方差
    seq_best = dp_seq[r][c]

    # 没有考虑方差（对比）
    loss_min = dp_loss.min()
    index = dp_loss.argmin()
    r_, c_ = index // (prop_quant + 1), index % (prop_quant + 1)
    prop_lossmin = tc.nansum(dp_w[r_][c_])/(~tc.isnan(dp_w[r][c])).sum()
    var_lossmin = dp_var[r_][c_]
    # seq_lossmin = dp_seq[r_][c_]

    selection["stride"] = stride
    selection["row"] = r
    selection["column"] = c
    selection["best_prop_loss_var"] = (prop_best.cpu().item(), loss_best.cpu().item(), var_best.cpu().item())  # 最佳loss及其方差
    selection["lossmin_prop_loss_var"] = (prop_lossmin.cpu().item(), loss_min.cpu().item(), var_lossmin.item())  # 最小loss及其方差
    selection["dp"] = dp_w.cpu().numpy()
    selection["weight_cls"] = weight_cls.cpu().numpy()

    if change:  # 改变了
        selection["selected"] = ~seq_best.cpu().numpy()
        selection["unselected"] = seq_best.cpu().numpy()
    else:  # 没改变
        selection["selected"] = seq_best.cpu().numpy()
        selection["unselected"] = ~seq_best.cpu().numpy()

    # 验证序列的正确性
    w = tc.sum(data[seq_best], axis=0)
    loss_ = get_loss_tc(w, val_proportion)
    assert abs(loss_.item() - selection["best_prop_loss_var"][1]) < 1e-6, "correct loss: {}, but: {}".format(selection["best_prop_loss_var"][1], loss_)

    return selection

def get_specifiction(data, selected, val_proportion):
    loss_cls = get_loss(data[selected], val_proportion, mode="acc")
    prop_cls = get_weight(data[selected])
    return prop_cls, loss_cls
    # 根据筛选的索引，统计最优选择的每个目标的比例和损失

def adjust_data(ori_data):
    # 物体损失计算
    # 多目标的优化难以仅依靠同台规划解决，因此我们在多目标中引入贪心，让loss大（数据量小）的数据有更多填充的可能
    loss = get_loss(ori_data, val_proportion)
    loss_adjust_idxs = np.argsort(loss, axis=-1)
    # idx_len = len(loss_adjust_idxs)
    # loss_adjust_idxs = np.concatenate([loss_adjust_idxs[idx_len//4 : idx_len*3//4],
    #                                    loss_adjust_idxs[idx_len : idx_len//4],
    #                                    loss_adjust_idxs[idx_len*3//4 : idx_len]])
    data = ori_data[loss_adjust_idxs]
    return data, loss_adjust_idxs

def partition_dp(df, val_proportion, stride=0.005, specific=True, use_var=False, loss_tolerance=None):
    """
    stride：float,手动设置步长
    loss_tolerance: 两种模式,use_var=True的时候有效
        int：容忍区间为最小loss的倍数（推荐）
        float：手动设置容忍区间
        None：不使用loss_tolerance，即不优先筛选loss在同时优化loss和方差，而是直接全部loss和方差同时计算优化结果
    """
    # 注意：df.values 数据中存有nan值，处理的时候应用np.nanxx()

    data, adjust_idxs = adjust_data(df.values)
    selection = get_best_select(data, val_proportion, stride, use_var, loss_tolerance)
    selected, unselected = adjust_idxs[selection["selected"]], adjust_idxs[selection["unselected"]]
    selected_paths = df.index[selected]
    unselected_paths = df.index[unselected]

    results = {}
    results["selected_paths"] = selected_paths  # 选中的路径
    results["unselected_paths"] = unselected_paths  # 未选中的路径
    results["best_prop_loss_var"] = selection["best_prop_loss_var"]  # 总的最好权重
    results["lossmin_prop_loss_var"] = selection["lossmin_prop_loss_var"]  # 总的最好loss
    if specific:
        prop_cls, loss_cls = get_specifiction(df.values, selected, val_proportion)
        results["prop_cls"] = prop_cls  # 每个类别的权重
        results["loss_cls"] = loss_cls  # 每个类别的loss
    return results

def multi_partition_dp(df_dict, val_proportion, stride=0.005, specific=True, use_var=False, loss_tolerance=None):
    """
    # 处理多个数据集
    :param df_dict: datafrrame
    :param val_proportion: 验证集划分比例(0~1)
    :param stride: 动态规划的loss步长
    :param specific: 是否保存划分后详细的每个类别的loss和权重
    :param use_var: 是否使用方差/标准差优化数据
    :param loss_tolerance: loss的容忍区间，即现筛选出容忍区间内的选择，在使用方差优化，仅在use_var=True时候有效
    :return:
    """
    selected_paths_all_work = []
    unselected_paths_all_work = []
    max_len = max([len(name) for name in names])

    for df_key in df_dict:  # 依次处理每个work
        df = df_dict[df_key]
        results = partition_dp(df, val_proportion, stride, specific, use_var, loss_tolerance)
        # 拼接三个work的路径
        selected_paths_all_work += results["selected_paths"].tolist()
        unselected_paths_all_work += results["unselected_paths"].tolist()
        # 输出每个work
        for k in results.keys():
            if "cls" in k:
                print("\n")
                print(">>>>>>>", k)
                for i, name in enumerate(names):
                    space = " " * (max_len - len(name))
                    print(name, space, ": ", results[k][i])
            else:
                print(">>>>>>>", k, ": ")
                print(results[k])
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
    val_proportion = 0.27  # 验证集比例
    scale = 0.1  # 缩放规模
    csv_root = pathlib.Path("./")  # collect_split_dataset.py的csv文件
    # train_root = pathlib.Path("/data/Zhong_Shui/zhongshui_pth/train")  # train保存路径
    # val_root = pathlib.Path("/data/Zhong_Shui/zhongshui_pth/val")  # val保存路径
    suffix = ".pth"  # suffix 文件后缀.txt,.pth或.npy
    torch.cuda.set_device("cuda:2")
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    csv_paths = csv_root.rglob("*point_num_ratio.csv")
    # csv_paths = csv_root.rglob("Work_4.csv")
    df_dict = {k: pd.read_csv(k, index_col=0) for k in csv_paths}
    # if not train_root.exists():
    #     train_root.mkdir(parents=True)
    # if not val_root.exists():
    #     val_root.mkdir(parents=True)

    selected_paths_all_work, unselected_paths_all_work = multi_partition_dp(df_dict,
                                                                            val_proportion,
                                                                            stride=0.01,
                                                                            specific=True,
                                                                            use_var=False,
                                                                            loss_tolerance=1.2)

    # pool = Pool(processes=32)
    # partial_save_train = functools.partial(save_dataset, save_root=train_root, scale=1, suffix=suffix)
    # partial_save_val = functools.partial(save_dataset, save_root=val_root, scale=1, suffix=suffix)
    # list(tqdm(pool.imap(partial_save_train, unselected_paths_all_work), total=len(unselected_paths_all_work)))
    # list(tqdm(pool.imap(partial_save_val, selected_paths_all_work), total=len(selected_paths_all_work)))
    # pool.close()
    # pool.join()
