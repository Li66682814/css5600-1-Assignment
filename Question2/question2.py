"""下载 Telco Customer Churn 数据并用决策树预测客户流失。"""

import csv  # DictReader 按列名读取客户记录，reader 逐行读取参数组合
import shutil  # which() 查找 curl 的可执行文件
import subprocess  # run() 调用系统的 curl 可执行程序
import sys  # stderr 用于输出下载或解压错误
from pathlib import Path  # Path 用来定位数据目录和文件
from tempfile import TemporaryDirectory  # 管理临时目录并在操作结束后清理
from typing import Any  # 标注包含多种值类型的关键字参数字典
from zipfile import BadZipFile, ZipFile, is_zipfile  # 检查 ZIP 并读取指定文件

import pandas as pd  # DataFrame 组织客户数据和输出表格，to_numeric() 转换数值列
from sklearn.compose import ColumnTransformer  # 给数值列和类别列指定不同的预处理
from sklearn.metrics import accuracy_score, confusion_matrix  # 计算准确率和分类计数
from sklearn.model_selection import (
    GridSearchCV,  # 比较参数组合的交叉验证成绩并重训最佳模型
    StratifiedKFold,  # 按流失类别比例划分交叉验证折
    train_test_split,  # 划分训练集和测试集
)
from sklearn.pipeline import Pipeline  # 把预处理与分类器串联起来
from sklearn.preprocessing import OrdinalEncoder  # 用整数表示类别，每个特征保留一列
from sklearn.tree import DecisionTreeClassifier  # 训练决策树分类器


# 数据划分和决策树使用各自的随机种子，可以独立修改。
SPLIT_PARAMS: dict[str, Any] = {
    "test_size": 0.20,  # 20% 用于测试，其余用于训练
    "random_state": 42,  # 固定训练集、测试集和交叉验证的随机划分
}

# 固定每组参数训练时的随机性，其余参与搜索的参数从 CSV 读取。
TREE_RANDOM_STATE = 42

DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "blastchar/telco-customer-churn"
)
ARCHIVE_NAME = "telco-customer-churn.zip"
CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PARAMETER_GRID_NAME = "DecisionTreeParameterGrid.csv"
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def download_archive(archive_path: Path) -> None:
    """调用 curl 下载 Kaggle 数据集，并保存通过 ZIP 检查的压缩包。"""
    curl = shutil.which("curl")
    if curl is None:
        raise FileNotFoundError("找不到 curl，请确认它已加入 PATH。")

    # 先写入同一数据目录下的临时文件，下载成功后再替换正式文件。
    # 这样中断时不会把未下载完成的内容留作下次运行的缓存。
    with TemporaryDirectory(dir=archive_path.parent) as temp_dir:
        temporary_archive = Path(temp_dir) / archive_path.name
        subprocess.run(
            [
                curl,
                "--location",  # 跟随 Kaggle 到文件存储地址的重定向
                "--fail",  # HTTP 错误使 curl 返回非零退出码
                "--silent",
                "--show-error",  # 隐藏进度条，保留失败信息
                "--connect-timeout", "10",
                "--max-time", "120",
                "--output", str(temporary_archive),
                DATASET_URL,
            ],
            check=True,
        )
        # 有些错误页面也会返回成功状态，所以还要检查响应确实是 ZIP。
        if not is_zipfile(temporary_archive):
            raise ValueError("Kaggle 返回的内容不是有效 ZIP，请检查数据集下载页面。")
        temporary_archive.replace(archive_path)


def extract_csv(archive_path: Path, csv_path: Path) -> None:
    """从压缩包中读取题目要求的 CSV，并保存到数据目录。"""
    with ZipFile(archive_path) as archive:
        # 只读取指定成员，输出路径由脚本决定。
        try:
            member = archive.getinfo(CSV_NAME)
        except KeyError as error:
            raise ValueError(f"ZIP 中找不到 {CSV_NAME}。") from error
        if member.file_size == 0:
            raise ValueError(f"ZIP 中的 {CSV_NAME} 为空。")

        # 读取完整个成员后才生成正式 CSV，避免把解压中断的文件识别成缓存。
        with TemporaryDirectory(dir=csv_path.parent) as temp_dir:
            temporary_csv = Path(temp_dir) / csv_path.name
            with archive.open(member) as source, temporary_csv.open("wb") as target:
                # 分块读取并写入字节，避免把整份 CSV 一次载入内存。
                while chunk := source.read(64 * 1024):
                    target.write(chunk)
            temporary_csv.replace(csv_path)


def prepare_data(data_dir: Path) -> Path:
    """优先使用本地 CSV，其次尝试本地 ZIP，最后才从 Kaggle 下载。"""
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / CSV_NAME
    archive_path = data_dir / ARCHIVE_NAME

    if csv_path.is_file() and csv_path.stat().st_size > 0:
        print("已找到 CSV，跳过下载。")
        return csv_path

    if archive_path.is_file():
        try:
            extract_csv(archive_path, csv_path)
        except (BadZipFile, ValueError):
            print("已有 ZIP 无法解压出目标 CSV，将重新下载。")
        else:
            print("已使用本地 ZIP 解压，跳过下载。")
            return csv_path

    print("正在使用 curl 从 Kaggle 下载 Telco Customer Churn 数据。", flush=True)
    download_archive(archive_path)
    extract_csv(archive_path, csv_path)
    print("下载并解压完成。")
    return csv_path


def run_download() -> Path:
    """准备数据并返回 CSV 路径，下载失败时用非零退出码结束。"""
    # 数据目录跟随脚本位置，与运行命令时的工作目录无关。
    data_dir = Path(__file__).resolve().parent / "Question2Data"
    try:
        csv_path = prepare_data(data_dir)
    except (OSError, BadZipFile, ValueError, subprocess.CalledProcessError) as error:
        print(f"Question 2 数据准备失败：{error}", file=sys.stderr)
        sys.exit(1)

    print(f"数据文件：{csv_path}")
    return csv_path


def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """读取客户数据，返回输入特征和流失标签。"""
    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = list(reader.fieldnames or ())
        data = pd.DataFrame(list(reader), columns=columns)

    # 数据集中总费用留空的客户入网时长均为 0 个月，按尚无累计费用处理。
    # strip() 去掉空格，使空字符串和仅含空格的字段都能被识别。
    blank_total_charges = data["TotalCharges"].str.strip().eq("")
    data["TotalCharges"] = data["TotalCharges"].mask(blank_total_charges, "0")

    # DictReader 读出的是文本，转换后决策树才能按数值大小划分。
    for feature in NUMERIC_FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")

    # Churn 是预测目标，customerID 只是编号。其余 19 列保留为输入特征。
    target = data["Churn"]
    if not target.isin(["Yes", "No"]).all():
        raise ValueError("Churn 列的标签必须为 Yes 或 No。")
    features = data.drop(columns=["customerID", "Churn"])
    return features, target


def build_model(features: pd.DataFrame) -> Pipeline:
    """构建包含单列类别编码和决策树的流水线。"""
    categorical_features = [
        str(feature) for feature in features.columns
        if feature not in NUMERIC_FEATURES
    ]
    # 数值列直接传入，每个类别列用 0、1、2 等整数编码，仍占一列。
    # 测试或预测时遇到训练中没见过的类别，用 -1 表示。
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    preprocessor = ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("categorical", encoder, categorical_features),
        ],
        verbose_feature_names_out=False,
    )
    classifier = DecisionTreeClassifier(random_state=TREE_RANDOM_STATE)
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def load_parameter_grid(csv_path: Path) -> list[dict[str, list[str | int | None]]]:
    """将 CSV 中的每一行读取为一组完整的决策树参数。"""
    parameter_grid: list[dict[str, list[str | int | None]]] = []
    columns = ["criterion", "max_depth", "min_samples_split", "min_samples_leaf"]
    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        if next(reader, []) != columns:
            raise ValueError(f"{csv_path.name} 的表头应为 {', '.join(columns)}。")

        for row in reader:
            if not row:
                continue
            try:
                criterion, max_depth, min_samples_split, min_samples_leaf = row
                # CSV 中的 None 表示让决策树按分裂条件生长，不额外限制深度。
                depth_text = max_depth.strip()
                depth = None if depth_text.casefold() == "none" else int(depth_text)
                # classifier__ 指向流水线中的分类器。每个列表只放本行的一个值，
                # GridSearchCV 因此只尝试 CSV 中实际存在的整行组合。
                parameter_grid.append(
                    {
                        "classifier__criterion": [criterion.strip()],
                        "classifier__max_depth": [depth],
                        "classifier__min_samples_split": [int(min_samples_split)],
                        "classifier__min_samples_leaf": [int(min_samples_leaf)],
                    }
                )
            except ValueError as error:
                raise ValueError(
                    f"{csv_path.name} 第 {reader.line_num} 行应包含分裂标准、"
                    "树深度（整数或 None）、最少分裂样本数和最少叶节点样本数（整数）。"
                ) from error

    if not parameter_grid:
        raise ValueError(f"{csv_path.name} 至少需要一行参数。")
    return parameter_grid


def search_best_model(x_train: pd.DataFrame, y_train: pd.Series, grid_path: Path) -> GridSearchCV:
    """在训练集上比较各组参数，再用完整训练集重训最佳模型。"""
    parameter_grid = load_parameter_grid(grid_path)
    validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_PARAMS["random_state"])
    # 每一折都重新拟合流水线，类别编码只使用该折的训练样本。
    search = GridSearchCV(
        build_model(x_train),
        param_grid=parameter_grid,
        scoring="accuracy",
        cv=validation,
        n_jobs=-1,  # 在可用的 CPU 核心上并行评估不同参数组合和验证折
        refit=True,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    return search


def print_search_result(search: GridSearchCV) -> None:
    """打印得分最高的参数组合及其交叉验证正确率。"""
    best_params = "，".join(
        f"{name.removeprefix('classifier__')}={value}" for name, value in search.best_params_.items()
    )
    print(f"\n已比较 {len(search.cv_results_['params'])} 组参数（{search.n_splits_} 折交叉验证）")
    print(f"最佳参数：{best_params}")
    print(f"最佳交叉验证平均正确率：{search.best_score_:.2%}")


def print_evaluation(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """打印数据划分占比、混淆矩阵和测试集正确率。"""
    class_order = ["Yes", "No"]
    total_count = len(y_train) + len(y_test)
    print(f"\n训练集占比：{len(y_train) / total_count:.1%}，测试集占比：{len(y_test) / total_count:.1%}")

    # 混淆矩阵的行对应真实类别，列对应预测类别，两者均按 Yes、No 排列。
    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions, labels=class_order)
    matrix_table = pd.DataFrame(matrix, index=class_order, columns=class_order)
    print("\n混淆矩阵（行是真实类别，列是预测类别）：")
    print(matrix_table.to_string())

    # 每类正确率用对角线上的正确预测数除以该行的真实样本数。
    test_accuracy = float(accuracy_score(y_test, predictions))
    print(f"\n测试集总正确率：{test_accuracy:.2%}")
    for index, label in enumerate(class_order):
        actual_count = int(matrix[index].sum())
        correct_count = int(matrix[index, index])
        if actual_count:
            print(f"实际为 {label} 的正确率：{correct_count / actual_count:.2%}（{correct_count}/{actual_count}）")
        else:
            print(f"实际为 {label} 的正确率：无该类测试样本")

    # 多数类基线把每个测试客户都预测为训练集中样本最多的类别。
    majority_class = y_train.value_counts().idxmax()
    baseline_accuracy = float(y_test.eq(majority_class).mean())
    print(f"多数类基线准确率（始终预测 {majority_class}）：{baseline_accuracy:.2%}")


def print_tree_details(model: Pipeline) -> None:
    """打印树的深度、叶节点数、非零特征重要性和根节点分支统计。"""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    class_names = [str(label) for label in classifier.classes_]

    print(f"\n决策树深度：{classifier.get_depth()}，叶节点数：{classifier.get_n_leaves()}")
    # 重要性表示该特征在树中带来的加权不纯度下降，数值按总和为 1 归一化。
    importance = pd.Series(classifier.feature_importances_, index=feature_names)
    importance = importance[importance > 0].sort_values(ascending=False)
    print("\n决策树使用的特征及其重要性：")
    print(importance.to_string(float_format=lambda value: f"{float(value):.4f}"))

    tree = classifier.tree_
    root_feature = int(tree.feature[0])
    if root_feature >= 0:
        threshold = float(tree.threshold[0])
        yes_index = class_names.index("Yes")
        print(f"\n根节点分裂：{feature_names[root_feature]}，阈值 {threshold:.2f}")
        # 类别列的阈值比较的是编码，显示训练时确定的类别与整数对应关系。
        categorical_slice = preprocessor.output_indices_["categorical"]
        if categorical_slice.start <= root_feature < categorical_slice.stop:
            encoder = preprocessor.named_transformers_["categorical"]
            categories = encoder.categories_[root_feature - categorical_slice.start]
            mapping = "，".join(f"{code}={category}" for code, category in enumerate(categories))
            print(f"  类别编码：{mapping}")
        for condition, child in [("<=", tree.children_left[0]), (">", tree.children_right[0])]:
            # 取出子节点的 Yes 类占比，得到该分支的训练样本流失率。
            class_values = tree.value[child, 0]
            churn_rate = float(class_values[yes_index] / class_values.sum())
            print(f"  {condition} {threshold:.2f}：{tree.n_node_samples[child]} 条，流失率 {churn_rate:.2%}")


def run_question2() -> None:
    """准备数据、训练模型，并打印评估结果和决策树摘要。"""
    csv_path = run_download()
    features, target = load_data(csv_path)

    # stratify 保留 Yes 和 No 的大致比例，测试集只用于最后的评估。
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, stratify=target, **SPLIT_PARAMS
    )
    search = search_best_model(x_train, y_train, csv_path.with_name(PARAMETER_GRID_NAME))
    print_search_result(search)
    model = search.best_estimator_
    print_evaluation(model, x_test, y_train, y_test)
    print_tree_details(model)


def main() -> None:
    run_question2()


if __name__ == "__main__":
    main()
