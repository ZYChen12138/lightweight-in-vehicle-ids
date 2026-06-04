# -*- coding: utf-8 -*-
"""
merge_preprocessing_two.py

把 preprocessing.ipynb + two.ipynb 合并为一个完整 py 文件。

默认输入：
    ./Pre_submit_S.csv

默认输出：
    ./S.csv
    ./S_timestamp.csv

最终 D_timestamp.csv 的列结构：
    a1~a7,
    b1~b9,
    c1~c9,
    d1, d2, d3, d4,
    1~16,
    SubClass

含义：
    a1~a7  : 当前帧与上一帧时间间隔 Duration 的 7 位数字展开
    b1~b9  : 同 ID 前一帧与同 ID 前前一帧时间间隔 Delta_1 的 9 位数字展开
    c1~c9  : 当前帧与同 ID 前一帧时间间隔 Delta_2 的 9 位数字展开
    d1~d3  : Arbitration_ID 的 3 位十六进制字符展开，并转为十进制数字
    d4     : DLC
    1~16   : DATA 8 字节拆成 16 个十六进制字符，并转为十进制数字
    SubClass: 多分类标签

类别映射：
    Flooding -> 0
    Fuzzy/Fuzzing -> 1
    Normal -> 2
    Replay -> 3
    Spoofing -> 4

重要修正：
1. 原来逐行 for 循环找 Prev_Timestamp_1/2，改为 groupby + shift，结果等价但更快；
2. Data 不足 8 字节时改为左侧补 00；
3. 不再对整个 DataFrame 先 applymap(str)，避免 Timestamp 字符串相减报错；
4. 计算 Delta_1/Delta_2 前保证 Timestamp 相关列为数值类型；
5. 时间间隔默认乘以 1e6 后四舍五入取整数：
   - 如果 Timestamp 单位是秒，TIME_SCALE=1e6 正确；
   - 如果 Timestamp 已经是微秒整数，请把 TIME_SCALE 改为 1.0。
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. 默认参数
# ============================================================

DEFAULT_INPUT_CSV = "Pre_submit_S.csv"
DEFAULT_INTERMEDIATE_CSV = "S.csv"
DEFAULT_OUTPUT_CSV = "S_timestamp.csv"

# 如果原始 Timestamp 单位是秒，乘 1e6 转成微秒。
# 如果原始 Timestamp 已经是微秒整数，把这里改成 1.0。
TIME_SCALE = 1e6

DURATION_WIDTH = 7
DELTA_WIDTH = 9


SUBCLASS_MAP = {
    "flooding": 0,
    "fuzzy": 1,
    "fuzzing": 1,
    "normal": 2,
    "replay": 3,
    "spoofing": 4,
}


# ============================================================
# 2. 工具函数
# ============================================================

def normalize_col_name(name):
    """去除列名前后空格，兼容 BOM。"""
    return str(name).replace("\ufeff", "").strip()


def find_column(df, candidates):
    """
    按候选名寻找列，大小写不敏感。
    """
    col_map = {normalize_col_name(c).lower(): c for c in df.columns}
    for cand in candidates:
        key = normalize_col_name(cand).lower()
        if key in col_map:
            return col_map[key]

    raise KeyError(f"找不到列名，候选={candidates}，当前列={list(df.columns)}")


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def parse_dlc(x):
    try:
        v = int(float(str(x).strip()))
    except Exception:
        v = 8
    return max(0, min(8, v))


def normalize_id_to_3hex(x):
    """
    Arbitration_ID 转成固定 3 位十六进制字符串。

    例：
        "1A2"  -> "1A2"
        "0x5A" -> "05A"
        "5A0"  -> "5A0"
    """
    if pd.isna(x):
        return "000"

    s = str(x).strip().upper()
    if s == "":
        return "000"

    try:
        if s.lower().startswith("0x"):
            v = int(s, 16)
        else:
            v = int(s, 16)
        return format(v, "03X")[-3:]
    except Exception:
        # 如果出现异常，尽量保留最后 3 位并补齐
        s = re.sub(r"[^0-9A-Fa-f]", "", s).upper()
        if s == "":
            return "000"
        return s[-3:].zfill(3)


def hex_char_to_int(ch):
    ch = str(ch).strip().upper()
    if ch == "":
        return 0
    return int(ch, 16)


def parse_data_left_pad_to_8(data, dlc):
    """
    按 DLC 取有效 DATA 字节，不足 8 字节时左侧补 00。

    例：
        DLC=3, Data="AA BB CC"
        -> ["00","00","00","00","00","AA","BB","CC"]
    """
    if pd.isna(data):
        tokens = []
    else:
        text = str(data).strip()
        tokens = re.split(r"[,\s]+", text) if text else []
        tokens = [t.strip().upper() for t in tokens if t.strip() != ""]

    dlc = parse_dlc(dlc)
    tokens = tokens[:dlc]

    # 只保留每个字节最后两位，补足两位
    clean_tokens = []
    for t in tokens:
        t = re.sub(r"[^0-9A-Fa-f]", "", t).upper()
        if t == "":
            t = "00"
        t = t[-2:].zfill(2)
        clean_tokens.append(t)

    clean_tokens = clean_tokens[-8:]
    padded = ["00"] * (8 - len(clean_tokens)) + clean_tokens
    return padded


def map_subclass(x):
    """
    SubClass 映射为 0~4。
    """
    if pd.isna(x):
        raise ValueError("SubClass 存在空值，无法映射。")

    s = str(x).strip()
    if s in ["0", "1", "2", "3", "4"]:
        return int(s)

    key = s.lower()
    if key in SUBCLASS_MAP:
        return SUBCLASS_MAP[key]

    raise ValueError(
        f"未知 SubClass={x}，支持 Flooding/Fuzzy/Fuzzing/Normal/Replay/Spoofing 或 0~4。"
    )


def format_interval_digits(values, width, scale=1e6):
    """
    时间间隔 -> 乘 scale -> 四舍五入 -> 非负整数 -> 固定宽度数字字符串。

    如果数字超过 width 位，保留最后 width 位，保证输出矩阵宽度固定。
    """
    arr = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    arr = np.rint(arr * scale).astype(np.int64)
    arr = np.maximum(arr, 0)

    out = []
    for v in arr:
        s = str(int(v))
        if len(s) > width:
            s = s[-width:]
        out.append(s.zfill(width))
    return pd.Series(out, index=values.index)


def split_fixed_string_column(series, width, prefix):
    """
    把固定宽度字符串列拆成多个数字列。

    例：
        "0001234" -> a1=0,a2=0,a3=0,a4=1,a5=2,a6=3,a7=4
    """
    s = series.astype(str).str.zfill(width).str[-width:]
    data = {}
    for i in range(width):
        data[f"{prefix}{i + 1}"] = s.str[i].astype(int)
    return pd.DataFrame(data, index=series.index)


# ============================================================
# 3. 第一步：对应 preprocessing.ipynb
# ============================================================

def stage1_build_intermediate(df_raw):
    """
    输入原始 Pre_submit_D.csv，输出相当于 D.csv 的中间 DataFrame。

    保留列：
        Timestamp, Arbitration_ID, DLC, Data, Class, SubClass, Duration,
        Prev_Timestamp_1, Prev_Timestamp_2
    """
    df = df_raw.copy()
    df.columns = [normalize_col_name(c) for c in df.columns]

    time_col = find_column(df, ["Timestamp", "时间戳", "time", "Time"])
    id_col = find_column(df, ["Arbitration_ID", "ID", "id", "can_id", "CAN_ID"])
    dlc_col = find_column(df, ["DLC", "dlc"])
    data_col = find_column(df, ["Data", "DATA", "data"])
    class_col = find_column(df, ["Class", "class", "CLASS"])
    subclass_col = find_column(df, ["SubClass", "subclass", "SUBCLASS", "subClass"])

    # 统一列名，减少后续混乱
    df = df.rename(columns={
        time_col: "Timestamp",
        id_col: "Arbitration_ID",
        dlc_col: "DLC",
        data_col: "Data",
        class_col: "Class",
        subclass_col: "SubClass",
    })

    # Timestamp 必须是数值
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).reset_index(drop=True)

    # Duration = 当前帧 Timestamp - 上一帧 Timestamp
    df["Duration"] = df["Timestamp"].diff().fillna(0)

    # 原 notebook 删除第一行，因为第一行 Duration 是人为 0
    df = df.drop(index=0).reset_index(drop=True)

    # Arbitration_ID 固定为 3 位十六进制字符串
    df["Arbitration_ID"] = df["Arbitration_ID"].apply(normalize_id_to_3hex)

    # 用 groupby + shift 得到同 ID 的前一帧、前前一帧时间戳
    # 与原来逐 ID、逐行循环的结果等价，但速度更快
    df["Prev_Timestamp_1"] = df.groupby("Arbitration_ID", sort=False)["Timestamp"].shift(1)
    df["Prev_Timestamp_2"] = df.groupby("Arbitration_ID", sort=False)["Timestamp"].shift(2)

    # 删除每个 ID 组里没有前两帧的样本
    df = df.dropna(subset=["Prev_Timestamp_1", "Prev_Timestamp_2"]).reset_index(drop=True)

    return df


# ============================================================
# 4. 第二步：对应 two.ipynb
# ============================================================

def stage2_build_final_matrix(df_mid):
    """
    输入 D.csv 中间 DataFrame，输出最终 D_timestamp.csv 的 DataFrame。
    """
    df = df_mid.copy()

    # 保证数值列类型正确，不要提前把整个 df 转成字符串
    for col in ["Timestamp", "Prev_Timestamp_1", "Prev_Timestamp_2", "Duration"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Timestamp", "Prev_Timestamp_1", "Prev_Timestamp_2", "Duration"]).reset_index(drop=True)

    # DATA 按 DLC 左侧补 00 到 8 字节
    padded_data = []
    for data_text, dlc in zip(df["Data"].values, df["DLC"].values):
        padded_data.append(parse_data_left_pad_to_8(data_text, dlc))
    df["Data"] = [" ".join(x) for x in padded_data]

    # d1 d2 d3：ID 三个十六进制字符展开为十进制数字
    ids = df["Arbitration_ID"].apply(normalize_id_to_3hex)
    df["d1"] = ids.str[0].apply(hex_char_to_int)
    df["d2"] = ids.str[1].apply(hex_char_to_int)
    df["d3"] = ids.str[2].apply(hex_char_to_int)

    # d4：DLC
    df["d4"] = df["DLC"].apply(parse_dlc).astype(int)

    # DATA 8 字节 -> 16 个十六进制字符 -> 十进制数字
    data_hex = df["Data"].str.replace(" ", "", regex=False).str.upper()

    for i in range(16):
        col_name = str(i + 1)
        df[col_name] = data_hex.str[i].apply(hex_char_to_int)

    # 计算两个同 ID 时间间隔
    # Delta_1 = 同 ID 前一帧时间戳 - 同 ID 前前一帧时间戳
    # Delta_2 = 当前帧时间戳 - 同 ID 前一帧时间戳
    df["Delta_1"] = df["Prev_Timestamp_1"] - df["Prev_Timestamp_2"]
    df["Delta_2"] = df["Timestamp"] - df["Prev_Timestamp_1"]

    # 时间间隔转成固定宽度数字字符串
    duration_str = format_interval_digits(df["Duration"], width=DURATION_WIDTH, scale=TIME_SCALE)
    delta1_str = format_interval_digits(df["Delta_1"], width=DELTA_WIDTH, scale=TIME_SCALE)
    delta2_str = format_interval_digits(df["Delta_2"], width=DELTA_WIDTH, scale=TIME_SCALE)

    # 拆成 a1~a7, b1~b9, c1~c9
    a_df = split_fixed_string_column(duration_str, width=DURATION_WIDTH, prefix="a")
    b_df = split_fixed_string_column(delta1_str, width=DELTA_WIDTH, prefix="b")
    c_df = split_fixed_string_column(delta2_str, width=DELTA_WIDTH, prefix="c")

    # 标签映射
    label_series = df["SubClass"].apply(map_subclass).astype(int)

    # 固定最终列顺序
    id_cols = ["d1", "d2", "d3", "d4"]
    data_cols = [str(i) for i in range(1, 17)]

    out_df = pd.concat(
        [
            a_df,
            b_df,
            c_df,
            df[id_cols].astype(int),
            df[data_cols].astype(int),
            label_series.rename("SubClass"),
        ],
        axis=1,
    )

    return out_df


# ============================================================
# 5. main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_CSV, help="输入原始 CSV，默认 Pre_submit_D.csv")
    parser.add_argument("--intermediate", type=str, default=DEFAULT_INTERMEDIATE_CSV, help="中间输出 CSV，默认 D.csv")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_CSV, help="最终输出 CSV，默认 D_timestamp.csv")
    parser.add_argument("--no_intermediate", action="store_true", help="不保存中间 D.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    intermediate_path = Path(args.intermediate)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path.resolve()}")

    print("=" * 80)
    print("读取原始文件")
    print("=" * 80)
    print("Input:", input_path.resolve())

    df_raw = pd.read_csv(input_path, encoding="utf-8-sig")
    print("原始数据 shape:", df_raw.shape)
    print("原始列名:", list(df_raw.columns))

    print("\n" + "=" * 80)
    print("Stage 1: 构造 Duration、同 ID 前两帧时间戳")
    print("=" * 80)

    df_mid = stage1_build_intermediate(df_raw)
    print("中间数据 shape:", df_mid.shape)
    print("中间列名:", list(df_mid.columns))

    if not args.no_intermediate:
        df_mid.to_csv(intermediate_path, index=False, encoding="utf-8-sig")
        print("已保存中间文件:", intermediate_path.resolve())

    print("\n" + "=" * 80)
    print("Stage 2: DATA 左补、ID/DATA 数字展开、时间间隔数字展开")
    print("=" * 80)

    out_df = stage2_build_final_matrix(df_mid)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("最终矩阵 shape:", out_df.shape)
    print("最终列名:", list(out_df.columns))
    print("已保存最终文件:", output_path.resolve())

    print("\n最终矩阵结构：")
    print("  a1~a7   : 7 维，当前帧与上一帧 Duration 的数字展开")
    print("  b1~b9   : 9 维，同 ID 前一帧 - 同 ID 前前一帧 Delta_1 的数字展开")
    print("  c1~c9   : 9 维，当前帧 - 同 ID 前一帧 Delta_2 的数字展开")
    print("  d1~d3   : 3 维，Arbitration_ID 三个十六进制字符转十进制")
    print("  d4      : 1 维，DLC")
    print("  1~16    : 16 维，DATA 8 字节拆为 16 个十六进制字符并转十进制")
    print("  SubClass: 1 维，多分类标签")
    print("  特征维度: 45，标签维度: 1，总列数: 46")

    print("\nSubClass 映射：")
    print("  Flooding -> 0")
    print("  Fuzzy/Fuzzing -> 1")
    print("  Normal -> 2")
    print("  Replay -> 3")
    print("  Spoofing -> 4")

    print("\n前 5 行：")
    print(out_df.head())


if __name__ == "__main__":
    main()
