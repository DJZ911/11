import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# 1. 数据加载
train_path = 'D:/python_dome/train.csv'
testA_path = 'D:/python_dome/testA.csv'
sample_submit_path = 'D:/python_dome/sample_submit.csv'

train_data = pd.read_csv(train_path)
testA_data = pd.read_csv(testA_path)


# 2. 数据预处理
def preprocess(df):
    # 处理时间特征
    df['issueDate'] = pd.to_datetime(df['issueDate'])
    df['issueDate_year'] = df['issueDate'].dt.year
    df['issueDate_month'] = df['issueDate'].dt.month
    df['issueDate_day'] = df['issueDate'].dt.day

    # 处理就业年限
    df['employmentLength'].replace({'< 1 year': '0', '10+ years': '10'}, inplace=True)
    df['employmentLength'] = df['employmentLength'].str.extract('(\d+)').astype(float)

    # 处理最早信用记录
    df['earliesCreditLine'] = df['earliesCreditLine'].str[-4:].astype(int)

    # 删除原始时间列
    df.drop(['issueDate'], axis=1, inplace=True)

    return df


train_data = preprocess(train_data)
testA_data = preprocess(testA_data)

# 3. 特征工程
# 选择特征列（根据比赛提供的字段表）
features = ['loanAmnt', 'term', 'interestRate', 'installment', 'grade', 'subGrade',
            'employmentLength', 'homeOwnership', 'annualIncome', 'verificationStatus',
            'purpose', 'regionCode', 'dti', 'delinquency_2years', 'ficoRangeLow',
            'ficoRangeHigh', 'openAcc', 'pubRec', 'pubRecBankruptcies', 'revolBal',
            'revolUtil', 'totalAcc', 'initialListStatus', 'applicationType',
            'earliesCreditLine', 'policyCode', 'issueDate_year', 'issueDate_month',
            'issueDate_day'] + [f'n{i}' for i in range(15)]

# 类别特征编码
cat_features = ['grade', 'subGrade', 'homeOwnership', 'verificationStatus',
                'purpose', 'initialListStatus', 'applicationType']

for col in cat_features:
    train_data[col] = train_data[col].astype('category')
    testA_data[col] = testA_data[col].astype('category')

# 4. 划分训练集和验证集
X_train = train_data[features]
y_train = train_data['isDefault']
X_test = testA_data[features]

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

# 5. 构建LightGBM模型
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42,
    'verbose': -1
}

lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=cat_features)

model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_val],
    num_boost_round=1000,
    early_stopping_rounds=50,
    verbose_eval=50
)

# 6. 模型评估
val_pred = model.predict(X_val, num_iteration=model.best_iteration)
val_auc = roc_auc_score(y_val, val_pred)
print(f'Validation AUC: {val_auc:.4f}')

# 7. 预测测试集并生成提交文件
test_pred = model.predict(X_test, num_iteration=model.best_iteration)

submit = pd.DataFrame({
    'id': testA_data['id'],
    'isDefault': test_pred
})

submit.to_csv('submission.csv', index=False)
print('Submission file saved as submission.csv')

# 8. 特征重要性分析
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importance()
}).sort_values('importance', ascending=False)

print('\nTop 10 important features:')
print(feature_importance.head(10))