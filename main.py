# train_repvgg_D_summary.py
# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 保持原代码：强制 CPU，如需 GPU 请注释掉这一行

import random
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from tqdm import tqdm


# ==================== 基本设置 ====================
DATA_PATH = r'.\S_timestamp.csv'
SAVE_DIR = './Results'
os.makedirs(SAVE_DIR, exist_ok=True)

SEEDS = [42]
EPOCHS = 200
BATCH_SIZE = 256
EARLY_STOPPING_PATIENCE = 15


# ==================== 类别名称 ====================
# 根据你前面的映射：
# Flooding -> 0, Fuzzy/Fuzzing -> 1, Normal -> 2, Replay -> 3, Spoofing -> 4
CLASS_NAME_MAP = {
    0: "Flooding",
    1: "Fuzzy",
    2: "Normal",
    3: "Replay",
    4: "Spoofing",
}


# ==================== 固定随机种子 ====================
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ==================== 内存保存最佳权重，不写 .h5 ====================
class BestWeightsByValAccuracy(tf.keras.callbacks.Callback):
    """
    只在内存中保存 val_accuracy 最优的权重，不写任何 .h5 文件。
    训练结束后调用 restore_best_weights(model) 恢复最优权重。
    """
    def __init__(self):
        super().__init__()
        self.best_val_accuracy = -np.inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get("val_accuracy")

        if current is None:
            return

        if current > self.best_val_accuracy:
            self.best_val_accuracy = float(current)
            self.best_weights = self.model.get_weights()
            self.best_epoch = int(epoch + 1)
            print(
                f"\nEpoch {epoch + 1:05d}: val_accuracy improved to "
                f"{self.best_val_accuracy:.6f}. Best weights kept in memory."
            )

    def restore_best_weights(self, model):
        if self.best_weights is not None:
            model.set_weights(self.best_weights)
            print(
                f"Restored best weights from epoch {self.best_epoch}, "
                f"best val_accuracy = {self.best_val_accuracy:.6f}"
            )
        else:
            print("[WARN] No best weights were recorded. Using current model weights.")


# ==================== 权重融合工具 ====================
def fuse_conv_bn(conv, bn):
    """Fuse Conv1D + BatchNorm into equivalent kernel & bias."""
    kernel = conv.kernel.numpy()  # [k, in_c, out_c]
    bias = conv.bias.numpy() if conv.use_bias else np.zeros(kernel.shape[-1], dtype=np.float32)

    gamma, beta, mean, var = bn.get_weights()
    std = np.sqrt(var + bn.epsilon)

    kernel = kernel * (gamma / std).reshape(1, 1, -1)
    bias = (bias - mean) * gamma / std + beta

    return kernel, bias


def fuse_identity_bn(bn, C):
    """Fuse identity(1×1 diag conv) + BatchNorm into kernel & bias."""
    k = np.zeros((1, C, C), dtype=np.float32)
    np.fill_diagonal(k[0], 1.0)

    b = np.zeros(C, dtype=np.float32)
    w = bn.get_weights()

    if len(w) != 4:
        return np.zeros_like(k), np.zeros_like(b)

    gamma, beta, mean, var = w
    std = np.sqrt(var + bn.epsilon)

    k = k * (gamma / std).reshape(1, 1, -1)
    b = (b - mean) * gamma / std + beta

    return k, b


# ==================== RepVGGBlock 定义 ====================
class RepVGGBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, deploy=False, **kw):
        super().__init__(**kw)
        self.deploy = deploy
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self._input_shape_for_deploy = None

        pad = 'same'

        if deploy:
            self.rbr_reparam = layers.Conv1D(
                filters,
                kernel_size,
                strides=strides,
                padding=pad,
                use_bias=True
            )
        else:
            self.rbr_identity = layers.BatchNormalization() if strides == 1 else None

            self.rbr_dense = layers.Conv1D(
                filters,
                kernel_size,
                strides=strides,
                padding=pad,
                use_bias=False
            )
            self.bn_dense = layers.BatchNormalization()

            self.rbr_1x1 = layers.Conv1D(
                filters,
                1,
                strides=strides,
                padding=pad,
                use_bias=False
            )
            self.bn_1x1 = layers.BatchNormalization()

        self.relu = layers.ReLU()

    def build(self, input_shape):
        self._input_shape_for_deploy = tuple(input_shape)
        super().build(input_shape)

    def call(self, x, training=None):
        if self.deploy:
            out = self.rbr_reparam(x)
        else:
            out = self.bn_dense(self.rbr_dense(x), training=training) + \
                  self.bn_1x1(self.rbr_1x1(x), training=training)

            if self.rbr_identity is not None and x.shape[-1] == self.filters:
                out += self.rbr_identity(x, training=training)

        return self.relu(out)

    def get_equivalent_kernel_bias(self):
        # 3×3 分支
        kd, bd = fuse_conv_bn(self.rbr_dense, self.bn_dense)

        # 1×1 分支，pad 到 3×3 中心
        k1, b1 = fuse_conv_bn(self.rbr_1x1, self.bn_1x1)
        pad = (self.kernel_size - 1) // 2

        k1_full = np.zeros_like(kd)
        k1_full[pad] = k1[0]

        # identity 分支
        if self.rbr_identity is not None and kd.shape[1] == self.filters:
            ki, bi = fuse_identity_bn(self.rbr_identity, self.filters)
            k_id_full = np.zeros_like(kd)
            k_id_full[pad] = ki[0]
        else:
            k_id_full = np.zeros_like(kd)
            bi = np.zeros_like(bd)

        return kd + k1_full + k_id_full, bd + b1 + bi

    def switch_to_deploy(self):
        k, b = self.get_equivalent_kernel_bias()

        self.rbr_reparam = layers.Conv1D(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding='same',
            use_bias=True
        )

        # 比原代码更稳，避免部分 Keras 版本中 self.input_shape 不存在
        input_shape = self._input_shape_for_deploy
        if input_shape is None:
            input_shape = (None, None, k.shape[1])

        self.rbr_reparam.build(input_shape)
        self.rbr_reparam.kernel.assign(k)
        self.rbr_reparam.bias.assign(b)

        for a in ('rbr_dense', 'bn_dense', 'rbr_1x1', 'bn_1x1', 'rbr_identity'):
            if hasattr(self, a):
                delattr(self, a)

        self.deploy = True

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "deploy": self.deploy
        })
        return cfg


# ==================== 构建 RepVGG 模型 ====================
def build_repvgg(inp_shape, n_cls, blocks=[2, 2, 2], filters=[4, 8, 16]):
    inp = layers.Input(shape=inp_shape)
    x = inp

    for stage, (n, f) in enumerate(zip(blocks, filters)):
        for i in range(n):
            s = 2 if (i == 0 and stage > 0) else 1
            x = RepVGGBlock(f, strides=s, deploy=False)(x)

    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(n_cls, activation='softmax')(x)

    return Model(inp, out)


# ==================== 统计工具 ====================
def class_name(label):
    return CLASS_NAME_MAP.get(int(label), f"Class-{int(label)}")


def one_vs_rest_accuracy(y_true, y_pred, label):
    y_true_bin = (y_true == label).astype(np.int32)
    y_pred_bin = (y_pred == label).astype(np.int32)
    return accuracy_score(y_true_bin, y_pred_bin)


def macro_accuracy_from_classes(y_true, y_pred, labels):
    accs = [one_vs_rest_accuracy(y_true, y_pred, lab) for lab in labels]
    return float(np.mean(accs))


def t_critical_95(n):
    if n <= 1:
        return np.nan
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df=n - 1))
    except Exception:
        # 常见 n 的 t 临界值，5 个 seed 时为 2.776
        lookup = {
            2: 12.706,
            3: 4.303,
            4: 3.182,
            5: 2.776,
            6: 2.571,
            7: 2.447,
            8: 2.365,
            9: 2.306,
            10: 2.262,
        }
        return lookup.get(n, 1.96)


def try_ttest_1samp(values, popmean):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return np.nan
    try:
        from scipy import stats
        return float(stats.ttest_1samp(values, popmean=popmean, nan_policy="omit").pvalue)
    except Exception:
        return np.nan


def summarize_values(values, chance_level=None):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    mean = float(np.mean(values)) if n > 0 else np.nan
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0

    if n > 1:
        tc = t_critical_95(n)
        half_width = float(tc * std / np.sqrt(n))
        ci_low = mean - half_width
        ci_high = mean + half_width
    else:
        ci_low = np.nan
        ci_high = np.nan

    if chance_level is None:
        p_value = np.nan
    else:
        p_value = try_ttest_1samp(values, popmean=chance_level)

    return {
        "mean": mean,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
    }


def fmt_float(x, digits=6):
    try:
        if np.isnan(x):
            return "N/A"
    except Exception:
        pass
    return f"{float(x):.{digits}f}"


def fmt_ci(low, high, digits=6):
    return f"[{fmt_float(low, digits)}, {fmt_float(high, digits)}]"


def clear_save_dir_keep_summary_only(save_dir):
    """
    保证 repvgg-D 目录中最终只有 summary.md。
    """
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for name in os.listdir(save_dir):
        p = os.path.join(save_dir, name)
        if os.path.isfile(p) or os.path.islink(p):
            os.remove(p)
        elif os.path.isdir(p):
            shutil.rmtree(p)


def write_summary_md(
    summary_path,
    data_path,
    save_dir,
    seeds,
    epochs,
    batch_size,
    labels_sorted,
    overall_records,
    class_records,
    confusion_records,
    params_list,
):
    overall_df = pd.DataFrame(overall_records)
    class_df = pd.DataFrame(class_records)
    confusion_df = pd.DataFrame(confusion_records)

    lines = []
    lines.append("# RepVGG-D Summary")
    lines.append("")
    lines.append("## Settings")
    lines.append("")
    lines.append(f"- Data path: `{data_path}`")
    lines.append(f"- Output directory: `{save_dir}`")
    lines.append(f"- Seed: `{seeds[0] if len(seeds) > 0 else 'N/A'}`")
    lines.append(f"- Epochs: `{epochs}`")
    lines.append(f"- Batch size: `{batch_size}`")
    lines.append(f"- Early stopping: monitor `val_accuracy`, patience = `{EARLY_STOPPING_PATIENCE}`")
    lines.append("- Split: stratified random 6:2:2")
    lines.append("- Model: RepVGG, blocks=[2,2,2], filters=[4,8,16]")
    lines.append("- Optimizer: Adam")
    lines.append("- Loss: sparse categorical crossentropy")
    lines.append("- Best model selection: `val_accuracy`, best weights kept in memory")
    lines.append("- Reported test metrics: evaluated using the best model selected by validation accuracy")
    lines.append("- Saved file policy: no `.h5` files are saved; only `summary.md` is kept in `./repvgg-D`")
    lines.append(f"- Parameters: `{int(params_list[0]) if len(params_list) > 0 else 'N/A'}`")
    lines.append("")

    lines.append("## Class Mapping")
    lines.append("")
    lines.append("| Label | Class |")
    lines.append("|---:|---|")
    for lab in labels_sorted:
        lines.append(f"| {int(lab)} | {class_name(lab)} |")
    lines.append("")

    lines.append("## Five-class Overall Metrics")
    lines.append("")
    lines.append("| Seed | Stopped Epoch | Best Epoch | Best Val Acc | Five-class Accuracy | Macro-Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Params |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in overall_df.iterrows():
        lines.append(
            f"| {int(row['seed'])} | "
            f"{int(row['stopped_epoch'])} | "
            f"{int(row['best_epoch'])} | "
            f"{row['best_val_accuracy']:.6f} | "
            f"{row['accuracy']:.6f} | "
            f"{row['macro_accuracy']:.6f} | "
            f"{row['macro_precision']:.6f} | "
            f"{row['macro_recall']:.6f} | "
            f"{row['macro_f1']:.6f} | "
            f"{int(row['params'])} |"
        )
    lines.append("")

    lines.append("## Per-class Attack Metrics")
    lines.append("")
    lines.append("| Class | Label | Accuracy | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in class_df.iterrows():
        lines.append(
            f"| {row['class_name']} | {int(row['label'])} | "
            f"{row['accuracy']:.6f} | "
            f"{row['precision']:.6f} | "
            f"{row['recall']:.6f} | "
            f"{row['f1']:.6f} |"
        )
    lines.append("")

    lines.append("## Confusion Matrix Records")
    lines.append("")
    lines.append("Each row means: true class = `true_label`, predicted class = `pred_label`, count = `count`.")
    lines.append("")
    lines.append("| True Label | True Class | Pred Label | Pred Class | Count |")
    lines.append("|---:|---|---:|---|---:|")
    for _, row in confusion_df.iterrows():
        lines.append(
            f"| {int(row['true_label'])} | {row['true_class']} | "
            f"{int(row['pred_label'])} | {row['pred_class']} | {int(row['count'])} |"
        )
    lines.append("")

    lines.append("## Raw Overall Records")
    lines.append("")
    lines.append("```csv")
    lines.append(overall_df.to_csv(index=False).strip())
    lines.append("```")
    lines.append("")

    lines.append("## Raw Per-class Records")
    lines.append("")
    lines.append("```csv")
    lines.append(class_df.to_csv(index=False).strip())
    lines.append("```")
    lines.append("")

    summary_path = os.path.abspath(summary_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ==================== 主程序 ====================
def main():
    clear_save_dir_keep_summary_only(SAVE_DIR)

    try:
        # ==================== 读取数据 ====================
        print("正在读取数据...")
        df = pd.read_csv(DATA_PATH)

        X_all = df.iloc[:, :-1].values.astype('float32')
        y_all = df.iloc[:, -1].values.astype('int32')

        labels_sorted = sorted(np.unique(y_all).tolist())
        num_classes = len(labels_sorted)

        # 保持原代码习惯，类别名用 0~n-1；summary 中再映射为攻击名
        class_names = [str(i) for i in labels_sorted]

        print("数据读取完成")
        print("DATA_PATH:", DATA_PATH)
        print("X_all shape:", X_all.shape)
        print("y_all shape:", y_all.shape)
        print("labels:", labels_sorted)
        print("num_classes:", num_classes)

        # ==================== 单个 seed 训练与测试 ====================
        overall_records = []
        class_records = []
        confusion_records = []
        params_list = []

        for seed in SEEDS:
            print("\n" + "=" * 70)
            print(f"开始训练 Seed = {seed}")
            print("=" * 70)

            tf.keras.backend.clear_session()
            set_global_seed(seed)

            # -------------------- 数据划分 --------------------
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_all,
                y_all,
                test_size=0.4,
                random_state=seed,
                stratify=y_all
            )

            X_val, X_test, y_val, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=0.5,
                random_state=seed,
                stratify=y_temp
            )

            # 增加通道维度
            X_train = np.expand_dims(X_train, -1)
            X_val = np.expand_dims(X_val, -1)
            X_test = np.expand_dims(X_test, -1)

            input_shape = X_train.shape[1:]

            print("X_train:", X_train.shape)
            print("X_val  :", X_val.shape)
            print("X_test :", X_test.shape)

            # -------------------- 创建模型 --------------------
            model = build_repvgg(input_shape, num_classes)

            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # 不保存任何 .h5 文件：只在内存中保存 val_accuracy 最优权重
            best_weight_callback = BestWeightsByValAccuracy()

            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=EARLY_STOPPING_PATIENCE,
                mode='max',
                restore_best_weights=False,
                verbose=1
            )

            # -------------------- 训练 --------------------
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[best_weight_callback, early_stop],
                verbose=1
            )

            stopped_epoch = len(history.history.get('loss', []))

            print(f"Seed {seed} 训练结束，未保存任何 .h5 文件。")
            print(f"Stopped epoch: {stopped_epoch}")
            print(f"Best epoch: {best_weight_callback.best_epoch}, best val_accuracy: {best_weight_callback.best_val_accuracy:.6f}")

            # -------------------- 恢复内存中的 val_accuracy 最优权重 --------------------
            train_m = model
            best_weight_callback.restore_best_weights(train_m)

            # -------------------- 切换为 deploy 模型，仅在内存中使用 --------------------
            for layer in train_m.layers:
                if isinstance(layer, RepVGGBlock) and not layer.deploy:
                    layer.switch_to_deploy()

            deploy_model = train_m
            params = deploy_model.count_params()
            params_list.append(params)

            print(f"Seed {seed} 部署模型参数量: {params}")

            # -------------------- 推理 --------------------
            preds = []
            steps = (len(X_test) + BATCH_SIZE - 1) // BATCH_SIZE

            for i in tqdm(range(steps), desc=f'Seed {seed} 推理进度'):
                start = i * BATCH_SIZE
                end = min((i + 1) * BATCH_SIZE, len(X_test))
                pred = deploy_model.predict(
                    X_test[start:end],
                    batch_size=end - start,
                    verbose=0
                )
                preds.append(pred)

            y_prob = np.vstack(preds)
            y_pred = np.argmax(y_prob, axis=1)

            # -------------------- 计算整体指标 --------------------
            acc = accuracy_score(y_test, y_pred)

            macro_acc = macro_accuracy_from_classes(y_test, y_pred, labels_sorted)

            macro_precision = precision_score(
                y_test,
                y_pred,
                labels=labels_sorted,
                average='macro',
                zero_division=0
            )
            macro_recall = recall_score(
                y_test,
                y_pred,
                labels=labels_sorted,
                average='macro',
                zero_division=0
            )
            macro_f1 = f1_score(
                y_test,
                y_pred,
                labels=labels_sorted,
                average='macro',
                zero_division=0
            )

            print(f"\nSeed {seed} 测试结果：")
            print(f"Accuracy        : {acc:.6f}")
            print(f"Macro-Accuracy  : {macro_acc:.6f}")
            print(f"Macro-Precision : {macro_precision:.6f}")
            print(f"Macro-Recall    : {macro_recall:.6f}")
            print(f"Macro-F1        : {macro_f1:.6f}")

            overall_records.append({
                'seed': seed,
                'stopped_epoch': int(stopped_epoch),
                'best_epoch': int(best_weight_callback.best_epoch),
                'best_val_accuracy': float(best_weight_callback.best_val_accuracy),
                'accuracy': float(acc),
                'macro_accuracy': float(macro_acc),
                'macro_precision': float(macro_precision),
                'macro_recall': float(macro_recall),
                'macro_f1': float(macro_f1),
                'params': int(params),
            })

            # -------------------- 计算每个攻击类别指标 --------------------
            per_precision = precision_score(
                y_test,
                y_pred,
                labels=labels_sorted,
                average=None,
                zero_division=0
            )
            per_recall = recall_score(
                y_test,
                y_pred,
                labels=labels_sorted,
                average=None,
                zero_division=0
            )
            per_f1 = f1_score(
                y_test,
                y_pred,
                labels=labels_sorted,
                average=None,
                zero_division=0
            )

            for idx, lab in enumerate(labels_sorted):
                class_records.append({
                    'seed': int(seed),
                    'label': int(lab),
                    'class_name': class_name(lab),
                    'accuracy': float(one_vs_rest_accuracy(y_test, y_pred, lab)),
                    'precision': float(per_precision[idx]),
                    'recall': float(per_recall[idx]),
                    'f1': float(per_f1[idx]),
                })

            # -------------------- 混淆矩阵记录 --------------------
            cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
            for i_true, true_lab in enumerate(labels_sorted):
                for i_pred, pred_lab in enumerate(labels_sorted):
                    confusion_records.append({
                        'seed': int(seed),
                        'true_label': int(true_lab),
                        'true_class': class_name(true_lab),
                        'pred_label': int(pred_lab),
                        'pred_class': class_name(pred_lab),
                        'count': int(cm[i_true, i_pred]),
                    })

            # 每个 seed 结束更新 summary.md
            write_summary_md(
                summary_path=os.path.join(SAVE_DIR, "summary.md"),
                data_path=DATA_PATH,
                save_dir=SAVE_DIR,
                seeds=SEEDS,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                labels_sorted=labels_sorted,
                overall_records=overall_records,
                class_records=class_records,
                confusion_records=confusion_records,
                params_list=params_list,
            )

        print("\n" + "=" * 70)
        print("Seed = 42 实验完成")
        print("=" * 70)
        print("summary.md 已保存：", os.path.abspath(os.path.join(SAVE_DIR, "summary.md")))

    finally:
        # 再清理一次 repvgg-D，确保除了 summary.md 没有其它文件
        summary_path = os.path.join(SAVE_DIR, "summary.md")
        for name in os.listdir(SAVE_DIR):
            p = os.path.join(SAVE_DIR, name)
            if os.path.abspath(p) == os.path.abspath(summary_path):
                continue
            if os.path.isfile(p) or os.path.islink(p):
                os.remove(p)
            elif os.path.isdir(p):
                shutil.rmtree(p)


if __name__ == "__main__":
    main()
