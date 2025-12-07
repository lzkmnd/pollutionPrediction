import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import logging
from scipy import stats
import os
import sys
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except ImportError:
    print("警告: 无法导入TensorFlow，LSTM模型将不可用。请运行 'pip install tensorflow' 来安装。")
    TF_AVAILABLE = False
else:
    TF_AVAILABLE = True

# 配置日志
logging.basicConfig(
    filename='feature_correlation.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='w'
)
SAVE_DIR = "/Volumes/personal_folder/code/csc/数据/result"
PREDICTIONS_DIR = os.path.join(SAVE_DIR, "predictions")
IMPORTANCE_DIR = os.path.join(SAVE_DIR, "importance_plots")
# 配置参数
TARGETS = ['PM2.5', 'PM10', 'O3']
BASE_FEATURES = ['T', 'WIND', 'SO2', 'P', 'R','NO2','CO']
N_LAGS = 24
ROLL_WINDOW = 12
WINDDIR_FEATURES = ['winddir_sin1', 'winddir_cos1', 'winddir_sin2', 'winddir_cos2']
# 支持的模型类型
AVAILABLE_MODELS = ['xgboost', 'random_forest', 'linear_regression', 'lstm']

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

    # 检查WINDDIR列是否存在
    if 'WINDDIR' not in df.columns:
        print("警告: 数据中未找到WINDDIR列，跳过风向编码")
        return df

    df['winddir_rad'] = np.deg2rad(df['WINDDIR'])
    df['winddir_sin1'] = np.sin(df['winddir_rad'])
    df['winddir_cos1'] = np.cos(df['winddir_rad'])
    df['winddir_sin2'] = np.sin(2 * df['winddir_rad'])
    df['winddir_cos2'] = np.cos(2 * df['winddir_rad'])
    return df.drop(columns=['WINDDIR', 'winddir_rad'])


def create_features(df, target_col):
    """创建完整特征集（集成特征筛选）"""
    # 动态确定可用的基础特征
    available_base_features = [f for f in BASE_FEATURES if f in df.columns]
    available_wind_features = [f for f in WINDDIR_FEATURES if f in df.columns]
    all_base_features = available_base_features + available_wind_features

    print(f"使用的特征列: {all_base_features}")

    if not all_base_features:
        raise ValueError("没有可用的基础特征列")

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


def create_lstm_model(input_shape):
    """创建LSTM模型"""
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))  # 回归输出
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def get_model(model_type='xgboost'):
    """根据模型类型返回相应的模型实例"""
    if model_type == 'xgboost':
        return XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            importance_type='gain',
            early_stopping_rounds=20,
            eval_metric='rmse'
        )
    elif model_type == 'random_forest':
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
    elif model_type == 'linear_regression':
        return LinearRegression()
    elif model_type == 'lstm':
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow未安装，无法使用LSTM模型。请安装TensorFlow后重试。")
        # LSTM模型需要在训练时根据输入形状创建
        return 'lstm_model_placeholder'
    else:
        raise ValueError(f"不支持的模型类型: {model_type}。支持的模型: {AVAILABLE_MODELS}")

def train_and_evaluate(X, y, model_type='xgboost'):
    """改进的模型训练流程（支持多种模型选择）"""
    tscv = TimeSeriesSplit(n_splits=3)
    
    # 特殊处理：线性回归没有特征重要性
    has_importances = model_type in ['xgboost', 'random_forest']
    is_lstm = model_type == 'lstm'

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
        if is_lstm:
            # LSTM需要3D输入形状 [samples, timesteps, features]
            # 为了简化，我们将特征重塑为24步的序列
            sequence_length = 24
            
            # 准备LSTM数据
            def create_sequences(X, y, seq_length):
                X_seq, y_seq = [], []
                for i in range(len(X) - seq_length):
                    X_seq.append(X[i:i+seq_length])
                    y_seq.append(y[i+seq_length])
                return np.array(X_seq), np.array(y_seq)
            
            # 创建序列数据
            X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, sequence_length)
            X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val.values, sequence_length)
            
            # 创建并训练LSTM模型
            input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
            model = create_lstm_model(input_shape)
            
            # 训练模型
            history = model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=50,
                batch_size=64,
                verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)]
            )
            
            # 预测
            pred = model.predict(X_val_seq).flatten()
            # 对齐预测结果和原始索引
            val_indices = np.arange(sequence_length, len(X_val_scaled))
            
            # 确保预测值和索引的长度匹配
            min_length = min(len(pred), len(val_indices))
            pred = pred[:min_length]
            val_indices = val_indices[:min_length]
            
            # 确保y_val的长度足够，避免索引越界
            actual_val = y_val.iloc[min(val_indices[0], len(y_val)-1):min(val_indices[-1]+1, len(y_val))]
            
            # 再次确保长度完全匹配
            final_length = min(len(pred), len(actual_val))
            pred = pred[:final_length]
            actual_val = actual_val.iloc[:final_length]
            
            # 创建预测结果DataFrame
            fold_pred = pd.DataFrame({
                'TIME': X_val.index[val_indices],
                'Actual': actual_val,
                'Predicted': pred,
                'Fold': f'fold_{fold + 1}'
            })
        else:
            # 获取模型
            model = get_model(model_type)
            
            # 训练模型
            if model_type == 'xgboost':
                model.fit(X_train_scaled, y_train,
                          eval_set=[(X_val_scaled, y_val)],
                          verbose=0)
            else:
                # RandomForest和LinearRegression不支持eval_set参数
                model.fit(X_train_scaled, y_train)
            
            # 收集预测结果
            pred = model.predict(X_val_scaled)
            fold_pred = pd.DataFrame({
                'TIME': X_val.index,
                'Actual': y_val,
                'Predicted': pred,
                'Fold': f'fold_{fold + 1}'
            })

        # 记录特征重要性（仅XGBoost和RandomForest支持）
        if has_importances:
            fold_importance = pd.DataFrame({
                'feature': X.columns,
                f'fold_{fold}': model.feature_importances_
            })
            feature_importances = pd.concat([feature_importances, fold_importance.set_index('feature')], axis=1)

        # 添加预测结果到总结果
        all_predictions = pd.concat([all_predictions, fold_pred])

        # 评估指标 - 对于LSTM使用调整后的actual_val
        if is_lstm:
            fold_results.append({
                'R2': r2_score(actual_val, pred),
                'RMSE': np.sqrt(mean_squared_error(actual_val, pred)),
                'MAE': mean_absolute_error(actual_val, pred)
            })
        else:
            fold_results.append({
                'R2': r2_score(y_val, pred),
                'RMSE': np.sqrt(mean_squared_error(y_val, pred)),
                'MAE': mean_absolute_error(y_val, pred)
            })

    # 处理特征重要性（仅XGBoost和RandomForest支持）
    if has_importances:
        feature_importances['avg_importance'] = feature_importances.mean(axis=1)
        feature_importances = feature_importances.sort_values('avg_importance', ascending=False)
    else:
        # 对于线性回归，创建一个空的重要性DataFrame
        feature_importances = pd.DataFrame(index=X.columns)
        feature_importances['avg_importance'] = 0.0

    return pd.DataFrame(fold_results), feature_importances, all_predictions


def plot_predictions(predictions, target, site_dir=None):
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
    if site_dir:
        plt.savefig(os.path.join(site_dir, f'{target}_prediction_plot.png'), dpi=300)
    else:
        plt.savefig(os.path.join(PREDICTIONS_DIR, f'{target}_prediction_plot.png'), dpi=300)
    plt.close()




def process_site_data(df, site_name, model_type='xgboost'):
    """处理单个站点的数据"""
    # 过滤出当前站点的数据
    if site_name == 'default':
        site_df = df.copy()
    else:
        site_df = df[df['SITENAME'] == site_name].copy()
    
    # 删除SITENAME列，因为它不是特征
    if 'SITENAME' in site_df.columns:
        site_df = site_df.drop(columns=['SITENAME'])
    
    # 设置时间索引
    site_df = site_df.set_index('TIME')
    site_df = site_df.ffill()
    
    print(f"处理站点: {site_name}")
    print(f"站点数据形状: {site_df.shape}")
    print(f"列名: {list(site_df.columns)}")
    
    site_df = encode_wind_direction(site_df)
    
    # 创建站点特定的输出目录
    site_predictions_dir = os.path.join(PREDICTIONS_DIR, site_name)
    site_importance_dir = os.path.join(IMPORTANCE_DIR, site_name)
    os.makedirs(site_predictions_dir, exist_ok=True)
    os.makedirs(site_importance_dir, exist_ok=True)

    final_metrics = {}

    for target in TARGETS:
        # 检查目标变量是否存在
        if target not in site_df.columns:
            print(f"警告: 目标变量 {target} 不在站点 {site_name} 的数据中，跳过")
            continue

        print(f"\n{'=' * 30}\n处理站点 {site_name} 的目标变量: {target}\n{'=' * 30}")

        try:
            # 特征工程
            full_data = create_features(site_df, target)
            X = full_data.drop(columns=['target'])
            y = full_data['target']

            print(f"特征数量: {X.shape[1]}, 样本数量: {X.shape[0]}")

            # 训练评估
            metrics, importances, predictions = train_and_evaluate(X, y, model_type)

            # 保存预测结果
            predictions.to_csv(os.path.join(site_predictions_dir, f'{target}_predictions.csv'), index=False)
            
            # 绘制预测图
            plot_predictions(predictions, target, site_predictions_dir)

            # 处理特征重要性
            total_importance = importances['avg_importance'].sum()
            importances['impact_pct'] = (importances['avg_importance'] / total_importance) * 100
            importances[['avg_importance', 'impact_pct']].to_csv(os.path.join(site_importance_dir, f'{target}_feature_impact.csv'))

            # 绘制重要性图
            plt.figure(figsize=(10, 8))
            importances['impact_pct'].head(20).sort_values().plot.barh(
                color='#2c7bb6',
                title=f'{target} - Top 20 Feature Importance (%)'
            )
            plt.xlabel('Importance Percentage')
            plt.tight_layout()
            plt.savefig(os.path.join(site_importance_dir, f'{target}_feature_importance.png'), dpi=300)
            plt.close()

            # 记录指标
            final_metrics[target] = {
                'R2': metrics['R2'].mean(),
                'RMSE': metrics['RMSE'].mean(),
                'MAE': metrics['MAE'].mean()
            }

            print(f"站点 {site_name} 的 {target} 处理完成 - R2: {final_metrics[target]['R2']:.4f}, "
                  f"RMSE: {final_metrics[target]['RMSE']:.4f}, MAE: {final_metrics[target]['MAE']:.4f}")

        except Exception as e:
            print(f"处理站点 {site_name} 的 {target} 时出错: {str(e)}")
            continue

    # 保存最终指标
    if final_metrics:
        pd.DataFrame(final_metrics).T.to_csv(os.path.join(site_predictions_dir, 'model_performance.csv'))
        print(f"\n站点 {site_name} 模型训练完成！所有结果已保存。")
    else:
        print(f"\n站点 {site_name} 没有成功处理任何目标变量。")
    return final_metrics


def main(data_path, model_type='xgboost'):
    """主流程 - 按站点分别处理"""
    # 检查文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    df = pd.read_excel(data_path, parse_dates=['TIME'])
    
    # 数据基本信息检查
    print("数据加载完成:")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print(f"时间范围: {df['TIME'].min()} 到 {df['TIME'].max()}")

    # 获取所有站点
    if 'SITENAME' in df.columns:
        sites = df['SITENAME'].unique()
        print(f"检测到站点: {sites}")
    else:
        # 如果没有SITENAME列，假设整个数据集就是一个站点
        sites = ['default']
        print("未检测到站点信息，将整个数据集作为一个站点处理")

    results = {}
    
    if 'SITENAME' in df.columns:
        # 按站点分别处理
        for site in sites:
            print(f"\n{'#' * 50}\n开始处理站点: {site}\n{'#' * 50}")
            site_data = df[df['SITENAME'] == site].copy()
            results[site] = process_site_data(site_data, site, model_type)
    else:
        # 处理无站点信息的数据
        print(f"\n{'#' * 50}\n开始处理默认站点\n{'#' * 50}")
        results['default'] = process_site_data(df, 'default', model_type)
    
    # 保存汇总结果
    summary = {}
    for site, metrics in results.items():
        for target, values in metrics.items():
            if target not in summary:
                summary[target] = {}
            summary[target][site] = values
    
    # 为每个污染物创建汇总报告
    for target in summary:
        if summary[target]:  # 只有当有数据时才保存
            summary_df = pd.DataFrame(summary[target]).T
            summary_df.to_csv(os.path.join(SAVE_DIR, f'{target}_all_sites_performance.csv'))
    
    print("\n所有站点处理完成！")
    return results




if __name__ == "__main__":
    excel_path = '/Volumes/personal_folder/code/csc/数据/虹口.xlsx'
    model_type = 'xgboost'  # 默认使用XGBoost模型
    
    # 允许通过命令行参数覆盖默认值
    if len(sys.argv) > 1:
        # 简单的参数解析，支持位置参数或--model参数
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == '--model' and i + 1 < len(sys.argv):
                model_type = sys.argv[i + 1].lower()
                i += 2
            else:
                # 第一个非--model参数作为excel路径
                excel_path = sys.argv[i]
                i += 1
    
    # 验证模型类型
    if model_type not in AVAILABLE_MODELS:
        print(f"错误: 不支持的模型类型 '{model_type}'")
        print(f"支持的模型: {', '.join(AVAILABLE_MODELS)}")
        sys.exit(1)
    
    print(f"使用模型: {model_type}")
    
    try:
        results = main(excel_path, model_type)
        print("\n各站点最终性能指标:")
        for site, metrics in results.items():
            print(f"\n站点: {site}")
            print(pd.DataFrame(metrics).T)
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        sys.exit(1)
