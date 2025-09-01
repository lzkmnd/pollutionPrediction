import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import logging
from scipy import stats
import os

# 配置日志
logging.basicConfig(
    filename='feature_correlation.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='w'
)
SAVE_DIR = "C:/Users/15800/Desktop/chengshi.xlsx"
PREDICTIONS_DIR = os.path.join(SAVE_DIR, "predictions")
IMPORTANCE_DIR = os.path.join(SAVE_DIR, "importance_plots")
# 配置参数
TARGETS = ['PM2.5', 'PM10', 'O3']
BASE_FEATURES = ['T', 'WIND', 'SO2', 'P', 'R','NO2','CO']
N_LAGS = 24
ROLL_WINDOW = 12
WINDDIR_FEATURES = ['winddir_sin1', 'winddir_cos1', 'winddir_sin2', 'winddir_cos2']

# 创建输出目录
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(IMPORTANCE_DIR, exist_ok=True)


def calculate_correlation(df, target_col):
    """计算特征与目标变量的相关性并记录日志"""
    corr_results = []
    pearson_corr = df.corr(method='pearson')[target_col].abs().sort_values(ascending=False)
    spearman_corr = df.corr(method='spearman')[target_col].abs().sort_values(ascending=False)

    for feat in df.columns:
        if feat != target_col:
            pearson_val = pearson_corr[feat]
            spearman_val = spearman_corr[feat]
            _, pvalue = stats.pearsonr(df[feat], df[target_col])
            corr_results.append({
                'Feature': feat,
                'Pearson': pearson_val,
                'Spearman': spearman_val,
                'P-value': pvalue,
                'Significant': pvalue < 0.05
            })

    corr_df = pd.DataFrame(corr_results).sort_values('Pearson', ascending=False)
    # corr_df.to_csv(f'{target_col}_correlation.csv', index=False)
    corr_df.to_csv(os.path.join(PREDICTIONS_DIR,f'{target_col}_correlation.csv'))

    logging.info(f"\n{'=' * 40}\n{target_col} 相关性分析结果 (Top 10):\n"
                 f"{corr_df.head(10).to_string(index=False)}\n"
                 f"显著相关特征数量: {sum(corr_df['Significant'])}/{len(corr_df)}\n"
                 f"最大皮尔逊系数: {corr_df['Pearson'].max():.3f}\n"
                 f"最大斯皮尔曼系数: {corr_df['Spearman'].max():.3f}")
    return corr_df


def encode_wind_direction(df):
    """优化风向编码"""
    df = df.copy()
    df['winddir_rad'] = np.deg2rad(df['WINDDIR'])
    df['winddir_sin1'] = np.sin(df['winddir_rad'])
    df['winddir_cos1'] = np.cos(df['winddir_rad'])
    df['winddir_sin2'] = np.sin(2 * df['winddir_rad'])
    df['winddir_cos2'] = np.cos(2 * df['winddir_rad'])
    return df.drop(columns=['WINDDIR', 'winddir_rad'])


def create_features(df, target_col):
    """创建完整特征集（集成特征筛选）"""
    all_base_features = BASE_FEATURES + WINDDIR_FEATURES

    # 特征生成
    feature_list = []
    for i in range(1, N_LAGS + 1):
        lag_features = df[all_base_features].shift(i).add_suffix(f'_lag{i}')
        feature_list.append(lag_features)

    target_lags = pd.concat([df[target_col].shift(i).rename(f'target_lag{i}')
                             for i in range(1, N_LAGS + 1)], axis=1)

    roll_stats = df[all_base_features].rolling(ROLL_WINDOW).agg(['mean', 'std'])
    roll_stats.columns = [f'{col}_roll{ROLL_WINDOW}_{stat}'
                          for col in all_base_features for stat in ['mean', 'std']]

    diff_features = pd.DataFrame()
    for var in all_base_features:
        diff_features[f'{var}_diff6h'] = df[var].diff(6)

    features = pd.concat([
        pd.concat(feature_list, axis=1),
        target_lags,
        roll_stats,
        diff_features
    ], axis=1)

    features = features.ffill().bfill()
    features['target'] = df[target_col]

    # 特征筛选
    corr_df = calculate_correlation(features.dropna(), 'target')
    selected_features = corr_df[
        (corr_df['Significant']) &
        (corr_df['Pearson'] > 0.15)
        ]['Feature'].tolist()

    if 'target' not in selected_features:
        selected_features.append('target')
    return features[selected_features]


def train_and_evaluate(X, y):
    """改进的模型训练流程（增加预测结果收集）"""
    tscv = TimeSeriesSplit(n_splits=3)
    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        importance_type='gain',
        early_stopping_rounds=20,
        eval_metric='rmse'
    )

    fold_results = []
    feature_importances = pd.DataFrame()
    all_predictions = pd.DataFrame()

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 标准化
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 训练模型
        model.fit(X_train_scaled, y_train,
                  eval_set=[(X_val_scaled, y_val)],
                  verbose=0)

        # 记录特征重要性
        fold_importance = pd.DataFrame({
            'feature': X.columns,
            f'fold_{fold}': model.feature_importances_
        })
        feature_importances = pd.concat([feature_importances, fold_importance.set_index('feature')], axis=1)

        # 收集预测结果
        pred = model.predict(X_val_scaled)
        fold_pred = pd.DataFrame({
            'TIME': X_val.index,
            'Actual': y_val,
            'Predicted': pred,
            'Fold': f'fold_{fold + 1}'
        })
        all_predictions = pd.concat([all_predictions, fold_pred])

        # 评估指标
        fold_results.append({
            'R2': r2_score(y_val, pred),
            'RMSE': np.sqrt(mean_squared_error(y_val, pred)),
            'MAE': mean_absolute_error(y_val, pred)
        })

    # 处理特征重要性
    feature_importances['avg_importance'] = feature_importances.mean(axis=1)
    feature_importances = feature_importances.sort_values('avg_importance', ascending=False)

    return pd.DataFrame(fold_results), feature_importances, all_predictions


def plot_predictions(predictions, target):
    """绘制预测结果时序图"""
    plt.figure(figsize=(15, 6))

    # 绘制实际值
    plt.plot(predictions['TIME'], predictions['Actual'],
             label='Actual', color='#2c7bb6', alpha=0.8, linewidth=1.5)

    # 分fold绘制预测值
    folds = predictions['Fold'].unique()
    colors = ['#d7191c', '#fdae61', '#abd9e9']
    for fold, color in zip(folds, colors):
        fold_data = predictions[predictions['Fold'] == fold]
        plt.scatter(fold_data['TIME'], fold_data['Predicted'],
                    label=fold, color=color, s=15, alpha=0.7)

    plt.title(f'{target} - Actual vs Predicted')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图片
    # plt.savefig(f'predictions/{target}_prediction_plot.png', dpi=300)
    plt.savefig(os.path.join(PREDICTIONS_DIR, f'{target}_prediction_plot.png'), dpi=300)
    plt.close()


def main(data_path):
    """主流程"""
    df = pd.read_excel(data_path, parse_dates=['TIME'], index_col='TIME').ffill()
    df = encode_wind_direction(df)
    predictions_dir = os.path.join(SAVE_DIR, "predictions")
    importance_dir = os.path.join(SAVE_DIR, "importance_plots")
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(IMPORTANCE_DIR, exist_ok=True)

    final_metrics = {}

    for target in TARGETS:
        print(f"\n{'=' * 30}\n处理目标变量: {target}\n{'=' * 30}")

        # 特征工程
        full_data = create_features(df, target)
        X = full_data.drop(columns=['target'])
        y = full_data['target']

        # 训练评估
        metrics, importances, predictions = train_and_evaluate(X, y)

        # 保存预测结果
        # predictions.to_csv(f'predictions/{target}_predictions.csv', index=False)
        predictions.to_csv(os.path.join(PREDICTIONS_DIR, f'{target}_predictions.csv'), index=False)
        # 绘制预测图
        plot_predictions(predictions, target)

        # 处理特征重要性
        total_importance = importances['avg_importance'].sum()
        importances['impact_pct'] = (importances['avg_importance'] / total_importance) * 100
        importances[['avg_importance', 'impact_pct']].to_csv(os.path.join(SAVE_DIR,f'importance_plots/{target}_feature_impact.csv'))

        # 绘制重要性图
        plt.figure(figsize=(10, 8))
        importances['impact_pct'].head(20).sort_values().plot.barh(
            color='#2c7bb6',
            title=f'{target} - Top 20 Feature Importance (%)'
        )
        plt.xlabel('Importance Percentage')
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR,f'importance_plots/{target}_feature_importance.png'), dpi=300)
        plt.close()

        # 记录指标
        final_metrics[target] = {
            'R2': metrics['R2'].mean(),
            'RMSE': metrics['RMSE'].mean(),
            'MAE': metrics['MAE'].mean()
        }

    # 保存最终指标
    # pd.DataFrame(final_metrics).T.to_csv('model_performance.csv')
    pd.DataFrame(final_metrics).T.to_csv(os.path.join(PREDICTIONS_DIR, 'model_performance.csv'))
    print("\n模型训练完成！所有结果已保存。")
    return final_metrics


if __name__ == "__main__":
    excel_path = 'C:/Users/15800/Desktop/chengshi1.xlsx'
    results = main(excel_path)
    print("\n最终性能指标:")
    print(pd.DataFrame(results).T)