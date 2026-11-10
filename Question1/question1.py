"""使用高斯朴素贝叶斯完成第 1 题的样本预测。"""

import csv  # csv 是 Python 标准库，DictReader 按列名读取每条记录
from pathlib import Path  # Path 用来定位与脚本同目录的数据文件

import numpy as np  # NumPy 用于数组计算，np.exp() 和 np.sqrt() 用于求高斯密度
import pandas as pd  # pandas 用于建表、筛选数据和打印结果
from sklearn.naive_bayes import GaussianNB  # scikit-learn 在 Python 中以 sklearn 导入。GaussianNB 是本题的分类器
from sklearn.pipeline import Pipeline  # Pipeline 将三个具名步骤按顺序串联
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # OneHotEncoder 为每个类别生成 0/1 列，StandardScaler 逐列做 Z-score 标准化


# 两种方法都返回明细表、预测类别和各类的后验概率。无法分类时类别为 None。
type PredictionResult = tuple[pd.DataFrame, str | None, dict[str, float]]


def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """读取 CSV，返回训练特征、训练标签和待预测特征。"""
    # 这四列是模型的输入特征，GoingOut 另作分类标签。
    feature_columns = ["Symptoms", "BodyTemp", "Vaccinated", "LockdownStatus"]

    # DictReader 根据首行的列名解析记录，pandas 再把这些记录组装成表格。
    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = list(reader.fieldnames or ())
        data = pd.DataFrame(list(reader), columns=columns)

    # DictReader 将空单元格读成空字符串。fillna() 也将缺失字段视为空标签。
    training_rows = data["GoingOut"].fillna("").ne("")
    # .loc[行条件, 列名] 同时选取需要的行和列。~ 会反转布尔条件。
    x_train = data.loc[training_rows, feature_columns]  # 训练样本的四个特征
    y_train = data.loc[training_rows, "GoingOut"]  # 对应的 Yes 或 No 标签
    x_new = data.loc[~training_rows, feature_columns]  # 待预测行的 GoingOut 为空

    # 本题只预测一条样本。空标签行数量不符时，停止并提示检查 CSV。
    if len(x_new) != 1:
        raise ValueError("Question1.csv 中 GoingOut 为空的记录必须恰好有一条。")
    return x_train, y_train, x_new


def calculate_gaussian_nb(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_new: pd.DataFrame,
) -> PredictionResult:
    """训练高斯朴素贝叶斯，返回似然密度表和预测结果。"""
    # Pipeline 按列表顺序执行各步骤，训练和预测都经过同一条流水线。
    # GaussianNB 接收稠密数组，所以编码器设置 sparse_output=False。
    model = Pipeline(
        [
            ("encoder", OneHotEncoder(sparse_output=False)),  # 将类别展开成 0/1 列
            ("scaler", StandardScaler()),  # 按训练列的均值和标准差做 Z-score 标准化
            ("classifier", GaussianNB()),  # 在每个类别内拟合各列的高斯分布
        ]
    )

    # fit() 从 x_train 学习编码规则和标准化参数，再将转换后的特征与 y_train
    # 交给 GaussianNB，估计各类的先验概率以及每列的类内均值和方差。
    model.fit(x_train, y_train)

    # named_steps 通过步骤名称取出已训练的组件，方便单独查看中间结果。
    encoder = model.named_steps["encoder"]
    scaler = model.named_steps["scaler"]

    # transform() 沿用训练时的类别顺序，将待预测样本编码成一行 0/1 值。
    encoded_sample = encoder.transform(x_new)
    # 用训练时的均值和标准差计算 z = (编码值 - 均值) / 标准差。
    # [0] 取出唯一一条待预测记录，得到与编码列一一对应的 Z-score。
    z_scores = scaler.transform(encoded_sample)[0]

    # theta_ 保存类内均值 μ，var_ 保存类内方差 σ²。两者的行是类别，列是编码特征。
    gaussian = model.named_steps["classifier"]
    # NumPy 将同一组 Z-score 分别与两类的均值相减，再逐项平方。
    squared_distances = (z_scores - gaussian.theta_) ** 2
    # 高斯密度为 exp(-(z - μ)² / (2σ²)) / sqrt(2πσ²)，分两步计算。
    densities = np.exp(-squared_distances / (2 * gaussian.var_))  # 计算分子
    densities /= np.sqrt(2 * np.pi * gaussian.var_)  # 除以分母，得到似然密度

    # 转置密度数组，让每行对应一个编码特征，每列对应一个预测类别。
    density_table = pd.DataFrame(
        densities.T,
        index=encoder.get_feature_names_out(),  # 取出训练得到的编码列名
        columns=gaussian.classes_,  # 按 GaussianNB 的实际类别顺序标记各列
    )
    density_table.index.name = "Encoded feature"
    # insert(0, ...) 将数值放到表格前面。astype(int) 将 0.0 和 1.0 显示为整数。
    density_table.insert(0, "Z-score", z_scores)
    density_table.insert(0, "Encoded value", encoded_sample[0].astype(int))

    # 把原始 x_new 传给流水线，由它自动完成编码、标准化和预测。
    prediction = str(model.predict(x_new)[0])  # 取出新样本的预测类别
    probabilities = model.predict_proba(x_new)[0]  # 取出该样本的各类后验概率

    # zip() 将 classes_ 中的类别和对应概率配对。strict=True 会检查两者长度一致。
    posterior_probabilities = {
        str(label): float(probability)
        for label, probability in zip(model.classes_, probabilities, strict=True)
    }
    return density_table, prediction, posterior_probabilities


def calculate_naive_bayes(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_new: pd.DataFrame,
) -> PredictionResult:
    """按手算公式复算离散朴素贝叶斯，返回条件概率表和预测结果。"""
    # 根据唯一的行索引取出待预测样本，保留四个原始特征及其列名。
    observed_values = x_new.loc[x_new.index[0]]
    manual_fractions = {}  # 以分数形式保存每类条件概率，供表格显示
    manual_scores: dict[str, float] = {}  # 保存每类乘上先验后的得分

    # unique() 取出不重复的类别。统一转成字符串，便于排序和作为字典键。
    class_labels: list[str] = [str(value) for value in y_train.unique()]
    class_labels.sort()
    for label in class_labels:
        group = x_train.loc[y_train == label]
        prior = len(group) / len(y_train)  # 类别先验等于该类记录数除以总记录数
        # eq() 按列名与待预测值比较，sum() 统计每一列中匹配的记录数。
        matching_counts = group.eq(observed_values).sum()
        conditional_probabilities = matching_counts / len(group)
        # 表格保留匹配数/类内总数的形式，数值计算使用上面的条件概率。
        manual_fractions[label] = matching_counts.astype(str) + f"/{len(group)}"
        # prod() 将四个条件概率相乘，再乘类别先验得到未归一化的得分。
        manual_scores[label] = prior * float(conditional_probabilities.prod())

    # 字典的键是类别，分数按原始特征名对齐成表格。
    manual_table = pd.DataFrame(manual_fractions)
    manual_table.index.name = "Feature"
    manual_table.insert(0, "Observed value", observed_values)

    # 各类得分的总和是贝叶斯公式的分母 P(X)。总和为零时无法做除法。
    total_score = sum(manual_scores.values())
    if total_score == 0:
        return manual_table, None, {}

    # max() 按字典中保存的得分找出最高分对应的类别。
    prediction = max(manual_scores, key=manual_scores.__getitem__)
    # 每类得分除以共同分母，得到该类的后验概率。
    posterior_probabilities = {
        label: score / total_score for label, score in manual_scores.items()
    }
    return manual_table, prediction, posterior_probabilities


def print_results(
    gaussian_result: PredictionResult, manual_result: PredictionResult
) -> None:
    """按顺序打印两种方法的明细、后验概率和预测对照。"""
    density_table, gaussian_prediction, gaussian_probabilities = gaussian_result
    manual_table, manual_prediction, manual_probabilities = manual_result

    # 只调整显示顺序，模型内部的类别顺序和计算结果保持原样。
    class_order = list(gaussian_probabilities)
    display_priority = {"Yes": 0, "No": 1}
    class_order.sort(key=lambda label: display_priority.get(label, 2))
    ordered_densities = density_table[["Encoded value", "Z-score", *class_order]]
    ordered_conditionals = manual_table[["Observed value", *class_order]]

    # to_string() 生成对齐的文本表格。float_format 只控制显示的小数位数。
    print("Conditional densities（GaussianNB 似然密度）")
    print(ordered_densities.to_string(float_format=lambda value: f"{float(value):.5f}"))
    print(f"\nGaussianNB 预测：GoingOut = {gaussian_prediction}")
    # .2% 将比例显示为保留两位小数的百分比。
    for label in class_order:
        probability = gaussian_probabilities[label]
        print(f"  {label} 的后验概率：{probability:.2%}")

    print("\nConditional probabilities（手算条件概率）")
    print(ordered_conditionals.to_string())
    if manual_prediction is None:
        print("手算公式的各类得分均为 0，无法计算后验概率。")
        return

    print(f"\n手算公式复算：GoingOut = {manual_prediction}")
    for label in class_order:
        probability = manual_probabilities[label]
        print(f"  {label} 的后验概率：{probability:.2%}")

    agreement = "一致" if gaussian_prediction == manual_prediction else "不同"
    print(f"预测类别是否一致：{agreement}")
    print("手算按原始特征统计频率，GaussianNB 对独热编码后的各列分别拟合高斯分布。")


def main() -> None:
    """依次读取数据、调用两种算法并输出结果。"""
    # __file__ 是当前脚本的路径。with_name() 换掉文件名，保留脚本所在目录。
    csv_path = Path(__file__).with_name("Question1.csv")
    x_train, y_train, x_new = load_data(csv_path)
    gaussian_result = calculate_gaussian_nb(x_train, y_train, x_new)
    manual_result = calculate_naive_bayes(x_train, y_train, x_new)
    print_results(gaussian_result, manual_result)


# 直接运行本脚本时执行 main()，被其他脚本导入时由调用者决定是否运行。
if __name__ == "__main__":
    main()
