import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


# 1. 数据加载与预处理 - 修复文件路径转义问题
def load_data():
    # 使用原始字符串或双反斜杠避免转义错误
    train = pd.read_csv(r'D:\python_dome\train.csv')  # 添加r前缀
    test = pd.read_csv(r'D:\python_dome\testA.csv')  # 添加r前缀

    full_data = pd.concat([train, test], axis=0, ignore_index=True)
    return full_data, train.shape[0]


# 2. 特征工程 - 修复employmentLength转换错误
def feature_engineering(df):
    # 处理日期特征
    df['issueDate'] = pd.to_datetime(df['issueDate'])
    df['issue_year'] = df['issueDate'].dt.year
    df['issue_month'] = df['issueDate'].dt.month
    df['issue_day'] = df['issueDate'].dt.day

    # 计算债务负担比率
    df['debt_burden'] = df['loanAmnt'] / (df['annualIncome'] + 1e-5)

    # 修复就业年限转换问题[4,6,11](@ref)
    # 先替换特殊值，再提取数字部分
    employment_map = {'< 1 year': '0', '10+ years': '10'}
    df['employmentLength'] = df['employmentLength'].replace(employment_map)

    # 提取数字部分并转换为浮点数[4,6](@ref)
    df['employmentLength'] = (
        df['employmentLength']
        .str.extract(r'(\d+)', expand=False)  # 提取数字部分
        .astype(float)
    )

    # 信用评分范围处理
    df['fico_avg'] = (df['ficoRangeLow'] + df['ficoRangeHigh']) / 2

    # 删除不需要的特征
    df.drop(['issueDate', 'ficoRangeLow', 'ficoRangeHigh'], axis=1, inplace=True)

    return df


# 3. 数据预处理
def preprocess_data(df):
    # 分类特征编码
    cat_features = ['grade', 'subGrade', 'homeOwnership', 'verificationStatus',
                    'purpose', 'regionCode', 'applicationType', 'initialListStatus']

    for col in cat_features:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes + 1  # 避免0值

    # 处理缺失值 - 使用更安全的to_numeric方法[1,6,11](@ref)
    for col in df.columns:
        if df[col].dtype == 'object':
            # 尝试转换为数值类型，失败则保留原值
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)

    # 特征缩放
    scaler = StandardScaler()
    num_features = ['loanAmnt', 'annualIncome', 'debt_burden', 'openAcc', 'revolBal', 'fico_avg']
    df[num_features] = scaler.fit_transform(df[num_features])

    return df


# 4. 模型训练与评估
def train_model(train_data, test_size=0.2):
    # 划分特征和目标变量
    X = train_data.drop(['isDefault', 'id'], axis=1)
    y = train_data['isDefault']

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # 创建LightGBM数据集
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # 模型参数
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }

    # 训练模型
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )

    # 验证集预测
    val_preds = model.predict(X_val)
    auc_score = roc_auc_score(y_val, val_preds)
    print(f'Validation AUC: {auc_score:.4f}')

    return model


# 5. 生成预测结果
def generate_submission(model, test_data, train_size):
    # 提取测试集
    test_df = test_data[test_data.index >= train_size].copy()
    test_df.reset_index(drop=True, inplace=True)

    # 准备测试特征
    X_test = test_df.drop(['id', 'isDefault'], axis=1)

    # 生成预测
    predictions = model.predict(X_test)

    # 创建提交文件
    submission = pd.DataFrame({
        'id': test_df['id'],
        'isDefault': predictions
    })

    return submission


# 主程序
if __name__ == "__main__":
    # 数据加载
    full_data, train_size = load_data()

    # 特征工程
    full_data = feature_engineering(full_data)

    # 数据预处理
    full_data = preprocess_data(full_data)

    # 分离训练集和测试集
    train_data = full_data.iloc[:train_size]

    # 模型训练
    model = train_model(train_data)

    # 生成预测结果
    submission = generate_submission(model, full_data, train_size)

    # 保存结果
    submission.to_csv(r'D:/python_dome/submission.csv', index=False)
    print('结果已保存至: D:/python_dome/submission.csv')
