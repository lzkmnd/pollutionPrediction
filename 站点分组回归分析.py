"""
大气污染物回归分析 - 站点分组版本 (Site-Based Analysis Edition)
基于原始分析脚本扩展，支持按站点分组分析功能

功能：
1. [智能检测] 自动识别数据中是否存在 SITENAME 字段
2. [灵活分组] 支持按站点输出独立分析结果，或全局分析
3. [参数控制] 支持通过参数指定分析特定站点
4. [独立输出] 各站点结果保存在独立文件夹，避免交叉污染
5. [完整继承] 保持原有分析逻辑的完整性和准确性
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import warnings
import platform
import json
from datetime import datetime
import gc  # 内存回收

# 基础与统计库
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 机器学习核心库
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# 可选库：贝叶斯优化
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    BAYESIAN_AVAILABLE = True
except ImportError:
    print("提示: 未检测到 scikit-optimize，将跳过贝叶斯优化使用默认参数。")
    BAYESIAN_AVAILABLE = False

# 可选库：SHAP 解释
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# 可选库：深度学习
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 屏蔽TF通知信息
except ImportError:
    TF_AVAILABLE = False

warnings.filterwarnings('ignore')

# --- 绘图字体统一设置 ---
def configure_fonts():
    """
    配置绘图字体，优先保证中文显示正常
    策略：
    1. 清除旧缓存并强制重建fontManager
    2. 强制加载系统中的宋体/Arial Unicode文件
    3. 中文字体优先：Songti SC放第一位，Times New Roman在后面
    """
    import matplotlib
    import matplotlib.font_manager as fm
    import os
    
    # 0. 尝试清除缓存并强制重建
    try:
        cache_dir = matplotlib.get_cachedir()
        cache_files = ['fontlist-v330.json', 'fontlist-v320.json', 'fontlist-v310.json']
        for cache_file in cache_files:
            json_path = os.path.join(cache_dir, cache_file)
            if os.path.exists(json_path):
                os.remove(json_path)
        
        import matplotlib.font_manager
        matplotlib.font_manager._load_fontmanager(try_read_cache=False)
    except:
        pass

    # macOS常见中文字体路径
    font_candidates = [
        '/System/Library/Fonts/Supplemental/Songti.ttc',
        '/Library/Fonts/Arial Unicode.ttf',
        '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
    ]
    
    loaded_fonts = []
    
    # 1. 加载中文字体文件
    for path in font_candidates:
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
                prop = fm.FontProperties(fname=path)
                font_name = prop.get_name()
                if font_name not in loaded_fonts:
                    loaded_fonts.append(font_name)
            except:
                pass
    
    # 2. 设置字体参数
    base_fonts = []
    if loaded_fonts:
        base_fonts.extend(loaded_fonts)
    base_fonts.append('Times New Roman')
    fallback_fonts = ['Songti SC', 'Arial Unicode MS', 'SimSun', 'PingFang SC', 'Heiti TC']
    base_fonts.extend(fallback_fonts)
    
    plt.rcParams['font.sans-serif'] = base_fonts
    plt.rcParams['font.serif'] = base_fonts
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    
    return base_fonts[0] if base_fonts else None

# 配置字体并获取首选字体
PRIMARY_FONT = configure_fonts()

# --- 全局图表样式配置 ---
def configure_plot_style():
    """
    配置全局图表样式，确保所有图表风格统一
    适用于论文发表的专业图表
    """
    # 字体大小统一
    plt.rcParams['font.size'] = 10          # 默认字体
    plt.rcParams['axes.titlesize'] = 14     # 标题
    plt.rcParams['axes.labelsize'] = 12     # 轴标签
    plt.rcParams['xtick.labelsize'] = 10    # x轴刻度
    plt.rcParams['ytick.labelsize'] = 10    # y轴刻度
    plt.rcParams['legend.fontsize'] = 9     # 图例
    
    # 线宽统一
    plt.rcParams['lines.linewidth'] = 1.5   # 默认线宽
    plt.rcParams['axes.linewidth'] = 1.0    # 坐标轴线宽
    plt.rcParams['grid.linewidth'] = 0.5    # 网格线宽
    
    # 网格样式
    plt.rcParams['grid.alpha'] = 0.3        # 网格透明度
    plt.rcParams['grid.linestyle'] = '--'   # 网格线型
    
    # 图表边距
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['figure.constrained_layout.use'] = False
    
    # 高质量输出
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    
    # 颜色循环（论文友好配色）
    # 使用ColorBrewer配色方案
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        '#2E86AB',  # 蓝色
        '#A23B72',  # 紫色
        '#F18F01',  # 橙色
        '#C73E1D',  # 红色
        '#6A994E',  # 绿色
    ])

configure_plot_style()

# 化学标识格式化函数
def format_pollutant_name(name):
    """
    将化学物质和污染物标识转换为正确的下标格式
    使用matplotlib数学模式以确保下标在所有字体下都能正确显示
    例如: SO2 -> SO$_2$, PM2.5 -> PM$_{2.5}$
    """
    # 创建映射字典 - 使用LaTeX数学模式
    replacements = {
        'SO2': 'SO$_2$',
        'NO2': 'NO$_2$',
        'PM10': 'PM$_{10}$',
        'PM2.5': 'PM$_{2.5}$',
        'O3': 'O$_3$',
        'CO': 'CO',  # CO保持不变
    }
    
    # 替换字符串中的污染物标识
    result = name
    for old, new in replacements.items():
        result = result.replace(old, new)
    
    return result

class SiteBasedAnalysisPipeline:
    """站点分组分析管道类"""
    def __init__(self, output_dir='./分析结果'):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_filename = None  # 保存数据文件名，用于命名输出文件夹
        
        # 处理路径问题 - 确保使用有效的目录
        try:
            # 首先尝试获取当前工作目录
            current_dir = os.getcwd()
        except:
            # 如果获取失败，使用脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 处理输出目录路径
        if output_dir == './分析结果':
            # 使用默认路径，确保在有效目录下
            output_dir = os.path.join(current_dir, '分析结果')
        elif not os.path.isabs(output_dir):
            # 如果是相对路径，转换为绝对路径
            output_dir = os.path.join(current_dir, output_dir)
        
        # 直接使用固定文件夹名，每次运行覆盖
        self.root_dir = os.path.join(output_dir, "站点分析")
        
        # 确保父目录存在
        parent_dir = os.path.dirname(self.root_dir)
        if parent_dir and not os.path.exists(parent_dir):
            try:
                os.makedirs(parent_dir, exist_ok=True)
            except Exception as e:
                print(f"警告: 无法创建父目录 {parent_dir}: {e}")
                # 回退到脚本所在目录
                output_dir = os.path.join(current_dir, '分析结果')
                self.root_dir = os.path.join(output_dir, f"站点分析_{self.timestamp}")
        
        try:
            os.makedirs(self.root_dir, exist_ok=True)
            print(f">>> 结果将保存至: {self.root_dir}")
        except Exception as e:
            # 如果仍然失败，使用临时目录
            import tempfile
            self.root_dir = tempfile.mkdtemp(prefix=f"pollution_analysis_{self.timestamp}_")
            print(f"警告: 无法创建指定目录，使用临时目录: {self.root_dir}")

    def create_output_structure(self, site_name=None):
        """
        创建输出目录结构
        site_name: 如果指定，则在站点子文件夹下创建
        无site_name时，使用数据文件名而非"全局分析"
        """
        if site_name:
            base_dir = os.path.join(self.root_dir, f"站点_{site_name}")
        else:
            # 使用数据文件名（去掉扩展名）
            if self.data_filename:
                base_name = os.path.splitext(self.data_filename)[0]
                base_dir = os.path.join(self.root_dir, base_name)
            else:
                base_dir = os.path.join(self.root_dir, "全局分析")
        
        paths = {
            'scatter': os.path.join(base_dir, "1_散点图"),
            'series': os.path.join(base_dir, "2_时序对比图"),
            'shap': os.path.join(base_dir, "3_SHAP特征分析"),
            'data': os.path.join(base_dir, "4_数据导出"),
            'logs': os.path.join(base_dir, "5_运行日志")
        }
        for p in paths.values():
            os.makedirs(p, exist_ok=True)
        
        return paths, os.path.join(paths['logs'], "详细运行日志.txt")

    def log(self, msg, log_file):
        """日志记录"""
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n")

    def calculate_iaqi(self, pollutant, concentration):
        """计算单个污染物的IAQI值（基于GB 3095-2012标准）"""
        # IAQI分段线性函数的断点表
        breakpoints = {
            'SO2': {
                'concentrations': [0, 50, 150, 475, 800, 1600, 2100, 2620],
                'iaqi_values': [0, 50, 100, 150, 200, 300, 400, 500]
            },
            'NO2': {
                'concentrations': [0, 40, 80, 180, 280, 565, 750, 940],
                'iaqi_values': [0, 50, 100, 150, 200, 300, 400, 500]
            },
            'PM10': {
                'concentrations': [0, 50, 150, 250, 350, 420, 500, 600],
                'iaqi_values': [0, 50, 100, 150, 200, 300, 400, 500]
            },
            'PM2.5': {
                'concentrations': [0, 35, 75, 115, 150, 250, 350, 500],
                'iaqi_values': [0, 50, 100, 150, 200, 300, 400, 500]
            },
            'CO': {
                'concentrations': [0, 2, 4, 14, 24, 36, 48, 60],
                'iaqi_values': [0, 50, 100, 150, 200, 300, 400, 500]
            },
            'O3': {
                'concentrations': [0, 100, 160, 215, 265, 800, 1000, 1200],
                'iaqi_values': [0, 50, 100, 150, 200, 300, 400, 500]
            }
        }
        
        if pollutant not in breakpoints:
            return np.nan
        
        bp = breakpoints[pollutant]
        conc_levels = bp['concentrations']
        iaqi_levels = bp['iaqi_values']
        
        # 边界检查
        if pd.isna(concentration) or concentration < 0:
            return np.nan
        
        # 线性插值计算IAQI
        for i in range(len(conc_levels) - 1):
            if conc_levels[i] <= concentration <= conc_levels[i + 1]:
                # IAQI = [(IAQI_high - IAQI_low) / (C_high - C_low)] * (C - C_low) + IAQI_low
                iaqi = ((iaqi_levels[i + 1] - iaqi_levels[i]) / 
                       (conc_levels[i + 1] - conc_levels[i])) * \
                       (concentration - conc_levels[i]) + iaqi_levels[i]
                return round(iaqi, 0)
        
        # 超出最高级别
        if concentration > conc_levels[-1]:
            return 500
        
        return 0

    def calculate_aqi_from_predictions(self, pred_df):
        """根据预测的污染物浓度计算AQI和首要污染物"""
        pollutants = ['SO2', 'NO2', 'PM10', 'PM2.5', 'CO', 'O3']
        
        # 计算各污染物的IAQI
        for pollutant in pollutants:
            pred_col = f'{pollutant}_预测值'
            if pred_col in pred_df.columns:
                iaqi_col = f'IAQI_{pollutant}'
                pred_df[iaqi_col] = pred_df[pred_col].apply(
                    lambda x: self.calculate_iaqi(pollutant, x)
                )
        
        # 找出所有IAQI列
        iaqi_cols = [col for col in pred_df.columns if col.startswith('IAQI_')]
        
        if iaqi_cols:
            # AQI = max(IAQI)
            pred_df['预测AQI'] = pred_df[iaqi_cols].max(axis=1)
            
            # 确定首要污染物（IAQI最大的污染物）
            def get_primary_pollutant(row):
                iaqi_values = {col.replace('IAQI_', ''): row[col] 
                             for col in iaqi_cols if pd.notna(row[col])}
                if not iaqi_values:
                    return 'N/A'
                max_iaqi = max(iaqi_values.values())
                if max_iaqi <= 50:  # AQI ≤ 50，无首要污染物
                    return '无'
                # 返回IAQI最大的污染物
                primary = [p for p, v in iaqi_values.items() if v == max_iaqi]
                return primary[0] if primary else 'N/A'
            
            pred_df['首要污染物'] = pred_df[iaqi_cols].apply(get_primary_pollutant, axis=1)
            
            # AQI等级
            def get_aqi_level(aqi):
                if pd.isna(aqi):
                    return 'N/A'
                if aqi <= 50:
                    return '优'
                elif aqi <= 100:
                    return '良'
                elif aqi <= 150:
                    return '轻度污染'
                elif aqi <= 200:
                    return '中度污染'
                elif aqi <= 300:
                    return '重度污染'
                else:
                    return '严重污染'
            
            pred_df['空气质量等级'] = pred_df['预测AQI'].apply(get_aqi_level)
        
        return pred_df

    def load_data(self, filepath, log_file):
        """加载与预处理数据"""
        self.log(f"正在加载数据: {os.path.basename(filepath)}", log_file)
        try:
            if filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath, parse_dates=['TIME'])
            else:
                df = pd.read_csv(filepath, parse_dates=['TIME'])
        except:
            try:
                df = pd.read_csv(filepath, parse_dates=['TIME'], encoding='gbk')
            except:
                df = pd.read_csv(filepath, encoding='gbk')
                if 'TIME' in df.columns:
                    df['TIME'] = pd.to_datetime(df['TIME'])
        
        # 检测是否存在SITENAME字段
        has_sitename = 'SITENAME' in df.columns
        
        if 'TIME' in df.columns:
            df['TIME'] = pd.to_datetime(df['TIME'])
            df = df.sort_values('TIME').reset_index(drop=True)
            
            # 插值填充缺失值（但不处理AQI列）
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            # 排除AQI列，因为AQI需要根据污染物重新计算，不能插值
            cols_to_interpolate = [col for col in numeric_cols if col not in ['AQI', 'IAQI']]
            if cols_to_interpolate:
                df[cols_to_interpolate] = df[cols_to_interpolate].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            
            # === 特征构造 ===
            if 'TEMP' in df.columns and 'R' in df.columns:
                df['Temp_R_Interaction'] = df['TEMP'] * df['R']
            
            # 季节特征处理
            if '季节' in df.columns:
                season_map = {'冬': 0, '春': 1, '夏': 2, '秋': 3}
                df['season_code'] = df['季节'].map(season_map)
            elif 'SEASON' in df.columns:
                season_map = {'冬': 0, '春': 1, '夏': 2, '秋': 3}
                df['season_code'] = df['SEASON'].map(season_map)
            elif '月' not in df.columns and 'month' not in df.columns:
                df['month'] = df['TIME'].dt.month
                def get_season(month):
                    if month in [12, 1, 2]:
                        return 0  # 冬
                    elif month in [3, 4, 5]:
                        return 1  # 春
                    elif month in [6, 7, 8]:
                        return 2  # 夏
                    else:
                        return 3  # 秋
                df['season_code'] = df['month'].apply(get_season)
        
        return df, has_sitename

    def feature_selection_physics(self, df, target, mode='strict', enable_physics_purification=False, log_file=None):
        """
        基于物理意义 + 统计学的双重特征筛选
        mode='strict': 适用于线性模型/LSTM (低VIF)
        mode='relaxed': 适用于树模型 (高VIF)
        enable_physics_purification: 是否开启物理净化 (防止数据泄露)
        """
        drop_cols = [target]
        
        # 全局排除结果指标（不能作为特征）
        result_indicators = ['AQI', 'VIS']  # AQI是综合指标，VIS是能见度（受污染物影响的结果）
        drop_cols.extend(result_indicators)
        
        # 排除非数值列 (SITENAME, TIME 等)
        df_num = df.select_dtypes(include=[np.number])
        
        # === 核心逻辑：物理净化 (防止数据泄露) ===
        if enable_physics_purification:
            if target == 'PM2.5':
                blacklist = ['PM10', 'VIS', 'CO']
                removed = []
                for b in blacklist:
                    if b in df_num.columns:
                        drop_cols.append(b)
                        removed.append(b)
                if mode == 'strict' and log_file:
                    self.log(f"    -> [防泄露] 预测PM2.5，强制剔除: {removed}", log_file)
                    
            elif target == 'PM10':
                if 'PM2.5' in df_num.columns:
                    drop_cols.append('PM2.5')
                    if mode == 'strict' and log_file:
                        self.log(f"    -> [防泄露] 预测PM10，强制剔除: ['PM2.5']", log_file)
            
            elif target == 'SO2':
                # SO2分析时可以考虑剔除相关污染物避免多重共线性
                so2_blacklist = ['NO2']  # 可根据实际情况调整
                removed = []
                for b in so2_blacklist:
                    if b in df_num.columns:
                        drop_cols.append(b)
                        removed.append(b)
                if mode == 'strict' and log_file and removed:
                    self.log(f"    -> [防泄露] 预测SO2，强制剔除: {removed}", log_file)
            
            elif target == 'NO2':
                # NO2分析时可以考虑剔除相关污染物
                no2_blacklist = ['SO2']  # 可根据实际情况调整
                removed = []
                for b in no2_blacklist:
                    if b in df_num.columns:
                        drop_cols.append(b)
                        removed.append(b)
                if mode == 'strict' and log_file and removed:
                    self.log(f"    -> [防泄露] 预测NO2，强制剔除: {removed}", log_file)
            
            elif target == 'CO':
                # CO分析时可以考虑剔除相关污染物
                co_blacklist = ['PM2.5', 'PM10']  # 可根据实际情况调整
                removed = []
                for b in co_blacklist:
                    if b in df_num.columns:
                        drop_cols.append(b)
                        removed.append(b)
                if mode == 'strict' and log_file and removed:
                    self.log(f"    -> [防泄露] 预测CO，强制剔除: {removed}", log_file)
        
        # 初始特征
        X = df_num.drop(columns=drop_cols, errors='ignore')
        feature_names = X.columns.tolist()
        
        # 设定阈值
        threshold_vif = 10.0 if mode == 'strict' else 100.0
        threshold_pearson = 0.90 if mode == 'strict' else 0.99
        
        # 1. 方差过滤
        sel = VarianceThreshold(threshold=0.01)
        sel.fit(X)
        curr_feats = [f for i, f in enumerate(feature_names) if sel.get_support()[i]]
        
        # 2. 皮尔逊去重
        df_curr = df_num[curr_feats]
        corr_matrix = df_curr.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold_pearson)]
        feat_p = [f for f in curr_feats if f not in to_drop]
        
        # 3. VIF 过滤
        df_vif = df_num[feat_p].copy()
        while True:
            try:
                vif_vals = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
                max_vif = max(vif_vals)
                if max_vif > threshold_vif:
                    drop_col = df_vif.columns[vif_vals.index(max_vif)]
                    df_vif = df_vif.drop(columns=[drop_col])
                else:
                    break
            except:
                break
        
        final_feats = df_vif.columns.tolist()
        
        # 4. 专家召回规则 - 根据污染物类型补充重要特征
        if target == 'O3':
            important = ['R', 'NO2', 'TEMP']
            for f in important:
                if f in df_num.columns and f not in final_feats:
                    final_feats.append(f)
        elif target == 'SO2':
            # SO2与气象条件、其他污染物相关
            important = ['TEMP', 'R', 'WIND', 'PM2.5']
            for f in important:
                if f in df_num.columns and f not in final_feats:
                    final_feats.append(f)
        elif target == 'NO2':
            # NO2与交通、气象条件相关
            important = ['TEMP', 'R', 'WIND', 'O3']
            for f in important:
                if f in df_num.columns and f not in final_feats:
                    final_feats.append(f)
        elif target == 'CO':
            # CO与交通、气象条件相关
            important = ['TEMP', 'WIND', 'NO2', 'PM2.5']
            for f in important:
                if f in df_num.columns and f not in final_feats:
                    final_feats.append(f)
        
        return df_num[final_feats].values, df_num[target].values.reshape(-1, 1), final_feats

    def optimize_model(self, X, y, model_name, log_file):
        """贝叶斯超参数优化"""
        if not BAYESIAN_AVAILABLE:
            return self.get_default_model(model_name)
        
        if model_name in ['xgboost', 'random_forest']:
            if model_name == 'xgboost':
                est = XGBRegressor(n_jobs=-1, random_state=42)
                space = {
                    'n_estimators': Integer(200, 600),
                    'max_depth': Integer(3, 10),
                    'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
                    'subsample': Real(0.6, 1.0)
                }
            else:
                est = RandomForestRegressor(n_jobs=-1, random_state=42)
                space = {
                    'n_estimators': Integer(100, 400),
                    'max_depth': Integer(5, 20),
                    'min_samples_split': Integer(2, 10)
                }
            try:
                opt = BayesSearchCV(est, space, n_iter=15, cv=3, n_jobs=-1, random_state=42, verbose=0)
                idx = np.random.choice(len(X), min(len(X), 300), replace=False)
                opt.fit(X[idx], y[idx].ravel())
                self.log(f"       [优化成功] {model_name} 最佳参数: {dict(opt.best_params_)}", log_file)
                return opt.best_estimator_
            except Exception as e:
                self.log(f"       [优化失败] {model_name}: {type(e).__name__}: {str(e)}", log_file)
                self.log(f"       [回退] 使用默认参数", log_file)
                return self.get_default_model(model_name)
        else:
            return self.get_default_model(model_name)

    def get_default_model(self, name):
        """获取默认模型"""
        if name == 'xgboost':
            return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42)
        elif name == 'random_forest':
            return RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
        elif name == 'linear_regression':
            return LinearRegression()
        return None

    def build_lstm(self, input_dim):
        """构建LSTM模型"""
        if not TF_AVAILABLE:
            return None
        model = Sequential([
            Input(shape=(1, input_dim)),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def analyze_single_dataset(self, df, site_name, targets, enable_physics_purification, base_output_dir):
        """
        分析单个数据集（全局或单个站点）
        df: 待分析的数据框
        site_name: 站点名称（None表示全局分析）
        """
        paths, log_file = self.create_output_structure(site_name)
        
        title = f"站点: {site_name}" if site_name else "全局分析"
        self.log(f"\n{'='*60}\n{title}\n{'='*60}", log_file)
        self.log(f"数据量: {len(df)} 条记录", log_file)
        
        # 提取数值列和时间轴
        df_num = df.select_dtypes(include=[np.number])
        time_axis = df['TIME'] if 'TIME' in df.columns else None
        
        summary = []
        # 用于存储所有污染物的预测结果
        all_predictions = {}
        
        for target in targets:
            if target not in df_num.columns:
                self.log(f"警告: 目标变量 {target} 不存在于数据中，跳过", log_file)
                continue
            
            self.log(f"\n{'='*50}\n正在分析目标: {target}\n{'='*50}", log_file)
            
            # 双轨制特征选择
            X_strict, y, feats_strict = self.feature_selection_physics(
                df, target, mode='strict', 
                enable_physics_purification=enable_physics_purification,
                log_file=log_file
            )
            X_relax, _, feats_relax = self.feature_selection_physics(
                df, target, mode='relaxed',
                enable_physics_purification=enable_physics_purification,
                log_file=None
            )
            
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            models = ['xgboost', 'random_forest', 'linear_regression']
            if TF_AVAILABLE:
                models.append('lstm')
            
            # 为每个目标选择最佳模型（用于AQI计算）
            best_model_name = None
            best_r2 = -np.inf
            best_predictions = None
            
            for m_name in models:
                # 根据模型选择特征集
                if m_name in ['xgboost', 'random_forest']:
                    X_curr, feats_curr = X_relax, feats_relax
                    mode_desc = "宽松特征集"
                else:
                    X_curr, feats_curr = X_strict, feats_strict
                    mode_desc = "严格特征集"
                
                self.log(f"  > 模型: {m_name:<18} | 策略: {mode_desc} (特征数:{len(feats_curr)})", log_file)
                
                # 参数优化
                base_model = self.optimize_model(X_curr, y, m_name, log_file)
                
                r2s, rmses, maes = [], [], []  # 添加MAE列表
                full_pred = np.zeros(len(y))
                
                last_model_for_shap = None
                last_X_train_for_shap = None
                
                # 交叉验证训练
                for t_idx, v_idx in kf.split(X_curr):
                    X_t, X_v = X_curr[t_idx], X_curr[v_idx]
                    y_t, y_v = y[t_idx], y[v_idx]
                    
                    sc_x, sc_y = StandardScaler(), StandardScaler()
                    X_t_s = sc_x.fit_transform(X_t)
                    X_v_s = sc_x.transform(X_v)
                    y_t_s = sc_y.fit_transform(y_t).flatten()
                    
                    if m_name == 'lstm':
                        model = self.build_lstm(X_curr.shape[1])
                        X_t_r = X_t_s.reshape(X_t_s.shape[0], 1, X_t_s.shape[1])
                        X_v_r = X_v_s.reshape(X_v_s.shape[0], 1, X_v_s.shape[1])
                        model.fit(X_t_r, y_t_s, epochs=40, batch_size=32, verbose=0,
                                callbacks=[EarlyStopping(monitor='loss', patience=5)])
                        p = model.predict(X_v_r, verbose=0).flatten()
                    else:
                        model = clone(base_model)
                        model.fit(X_t_s, y_t_s)
                        p = model.predict(X_v_s)
                        last_model_for_shap = model
                        last_X_train_for_shap = X_t_s
                    
                    p_real = sc_y.inverse_transform(p.reshape(-1, 1)).flatten()
                    full_pred[v_idx] = p_real
                    
                    r2s.append(r2_score(y_v, p_real))
                    rmses.append(np.sqrt(mean_squared_error(y_v, p_real)))
                    maes.append(mean_absolute_error(y_v, p_real))  # 添加MAE计算
                
                avg_r2 = np.mean(r2s)
                avg_rmse = np.mean(rmses)
                avg_mae = np.mean(maes)  # 计算平均MAE
                self.log(f"    评估: R2={avg_r2:.4f} | RMSE={avg_rmse:.4f} | MAE={avg_mae:.4f}", log_file)
                
                summary.append({
                    '站点': site_name if site_name else '全局',
                    '目标': target,
                    '模型': m_name,
                    'R2': avg_r2,
                    'RMSE': avg_rmse,
                    'MAE': avg_mae,  # 添加MAE
                    '特征': ",".join(feats_curr)
                })
                
                # 记录最佳模型的预测结果（用于AQI计算）
                if avg_r2 > best_r2:
                    best_r2 = avg_r2
                    best_model_name = m_name
                    best_predictions = full_pred.copy()
                
                # 输出结果
                self.plot_results(time_axis, y.flatten(), full_pred, target, m_name, avg_r2, avg_rmse, avg_mae, paths)
                self.save_pred_data(time_axis, y.flatten(), full_pred, target, m_name, paths)
                
                # SHAP 分析
                if SHAP_AVAILABLE and last_model_for_shap and m_name != 'lstm':
                    self.run_shap(last_model_for_shap, last_X_train_for_shap, feats_curr, target, m_name, paths)
                    self.calc_importance(last_model_for_shap, feats_curr, target, m_name, paths)
            
            # 存储该污染物的最佳预测结果
            if best_predictions is not None:
                all_predictions[target] = {
                    '真实值': y.flatten(),
                    '预测值': best_predictions,
                    '最佳模型': best_model_name,
                    'R2': best_r2
                }
        
        # === 新增: AQI整合分析 ===
        if all_predictions:
            self.log(f"\n{'='*60}\n整合预测结果并计算AQI\n{'='*60}", log_file)
            self.generate_aqi_integration_report(all_predictions, time_axis, paths, log_file, site_name)
        
        # 保存当前站点/全局的汇总结果
        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(paths['data'], "结果汇总.csv")
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        self.log(f"\n>>> 分析完成！结果已保存", log_file)
        
        return summary

    def generate_aqi_integration_report(self, all_predictions, time_axis, paths, log_file, site_name):
        """生成AQI整合报告"""
        self.log("正在生成AQI整合预测表...", log_file)
        
        # 创建基础数据框
        n_samples = len(list(all_predictions.values())[0]['真实值'])
        
        # 构建整合数据框
        integrated_df = pd.DataFrame()
        
        # 添加时间列
        if time_axis is not None:
            integrated_df['时间'] = time_axis
        else:
            integrated_df['序号'] = range(n_samples)
        
        # 添加站点信息
        if site_name:
            integrated_df['站点'] = site_name
        
        # 添加各污染物的真实值和预测值
        for pollutant, pred_data in all_predictions.items():
            integrated_df[f'{pollutant}_真实值'] = pred_data['真实值']
            integrated_df[f'{pollutant}_预测值'] = pred_data['预测值']
            integrated_df[f'{pollutant}_模型'] = pred_data['最佳模型']
            integrated_df[f'{pollutant}_R2'] = pred_data['R2']
        
        # 计算预测AQI（基于预测值）
        self.log("正在计算预测AQI和首要污染物...", log_file)
        integrated_df = self.calculate_aqi_from_predictions(integrated_df)
        
        # 计算真实AQI（优先使用原始数据，没有才计算）
        if 'AQI' in integrated_df.columns:
            # 原始数据中已有AQI，直接使用
            self.log("使用原始数据中的AQI作为真实AQI", log_file)
            integrated_df['真实AQI'] = integrated_df['AQI']
            # 如果有首要污染物和空气质量等级，也使用原始值
            if '首要污染物' in integrated_df.columns:
                integrated_df['真实首要污染物'] = integrated_df['首要污染物']
            if '空气质量等级' in integrated_df.columns:
                integrated_df['真实空气质量等级'] = integrated_df['空气质量等级']
        else:
            # 原始数据没有AQI，根据真实污染物浓度计算
            self.log("原始数据没有AQI，根据真实污染物浓度计算", log_file)
            true_df_for_aqi = pd.DataFrame()
            for pollutant in all_predictions.keys():
                true_df_for_aqi[f'{pollutant}_预测值'] = integrated_df[f'{pollutant}_真实值']
            true_df_with_aqi = self.calculate_aqi_from_predictions(true_df_for_aqi)
            
            if '预测AQI' in true_df_with_aqi.columns:
                integrated_df['真实AQI'] = true_df_with_aqi['预测AQI']
                integrated_df['真实首要污染物'] = true_df_with_aqi['首要污染物']
                integrated_df['真实空气质量等级'] = true_df_with_aqi['空气质量等级']
        
        # 计算AQI预测误差
        if '真实AQI' in integrated_df.columns and '预测AQI' in integrated_df.columns:
            integrated_df['AQI预测误差'] = integrated_df['预测AQI'] - integrated_df['真实AQI']
            integrated_df['AQI预测误差率(%)'] = (integrated_df['AQI预测误差'] / integrated_df['真实AQI'] * 100).round(2)
        
        # 保存整合预测表
        aqi_report_path = os.path.join(paths['data'], "AQI整合预测表.csv")
        integrated_df.to_csv(aqi_report_path, index=False, encoding='utf-8-sig')
        self.log(f"AQI整合预测表已保存: {aqi_report_path}", log_file)
        
        # 生成AQI预测统计摘要
        if '预测AQI' in integrated_df.columns and '真实AQI' in integrated_df.columns:
            aqi_mae = mean_absolute_error(integrated_df['真实AQI'], integrated_df['预测AQI'])
            aqi_rmse = np.sqrt(mean_squared_error(integrated_df['真实AQI'], integrated_df['预测AQI']))
            aqi_r2 = r2_score(integrated_df['真实AQI'], integrated_df['预测AQI'])
            
            # 首要污染物预测准确率
            primary_accuracy = (integrated_df['首要污染物'] == integrated_df['真实首要污染物']).sum() / len(integrated_df) * 100
            
            # 等级预测准确率
            level_accuracy = (integrated_df['空气质量等级'] == integrated_df['真实空气质量等级']).sum() / len(integrated_df) * 100
            
            self.log(f"\nAQI预测性能:", log_file)
            self.log(f"  MAE: {aqi_mae:.2f}", log_file)
            self.log(f"  RMSE: {aqi_rmse:.2f}", log_file)
            self.log(f"  R²: {aqi_r2:.4f}", log_file)
            self.log(f"  首要污染物预测准确率: {primary_accuracy:.2f}%", log_file)
            self.log(f"  空气质量等级预测准确率: {level_accuracy:.2f}%", log_file)
            
            # 保存AQI统计摘要
            aqi_stats = {
                '站点': [site_name if site_name else '全局'],
                'AQI_MAE': [aqi_mae],
                'AQI_RMSE': [aqi_rmse],
                'AQI_R2': [aqi_r2],
                '首要污染物准确率(%)': [primary_accuracy],
                '空气质量等级准确率(%)': [level_accuracy],
                '数据量': [len(integrated_df)]
            }
            aqi_stats_df = pd.DataFrame(aqi_stats)
            aqi_stats_path = os.path.join(paths['data'], "AQI预测统计.csv")
            aqi_stats_df.to_csv(aqi_stats_path, index=False, encoding='utf-8-sig')
            
            # 绘制AQI预测对比图
            self.plot_aqi_comparison(integrated_df, paths, site_name)
        
        self.log("AQI整合分析完成！", log_file)

    def plot_aqi_comparison(self, integrated_df, paths, site_name):
        """绘制AQI预测对比图（优化版）"""
        if '预测AQI' not in integrated_df.columns or '真实AQI' not in integrated_df.columns:
            return
        
        from scipy import stats
        
        # 1. AQI时序对比图
        fig, ax = plt.subplots(figsize=(12, 5))
        if '时间' in integrated_df.columns:
            ax.plot(integrated_df['时间'], integrated_df['真实AQI'], 
                    'k-', alpha=0.6, label='Observed AQI', linewidth=1.5)
            ax.plot(integrated_df['时间'], integrated_df['预测AQI'], 
                    'r--', alpha=0.8, label='Predicted AQI', linewidth=1.5)
            plt.gcf().autofmt_xdate()
        else:
            ax.plot(integrated_df['真实AQI'], 'k-', alpha=0.6, label='Observed AQI', linewidth=1.5)
            ax.plot(integrated_df['预测AQI'], 'r--', alpha=0.8, label='Predicted AQI', linewidth=1.5)
        
        # 简化标题
        title = f"AQI-{site_name}" if site_name else "AQI"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('AQI', fontsize=12)
        ax.set_xlabel('Time', fontsize=12) if '时间' in integrated_df.columns else ax.set_xlabel('Index', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(paths['series'], "AQI_时序对比.png"), dpi=300, bbox_inches='tight')
        plt.close('all')  # 关闭所有图表释放内存
        
        # 注意：AQI图表已简化，只保留时序图

    def run(self, data_path, targets=['PM2.5', 'PM10', 'O3', 'SO2', 'NO2', 'CO'], 
            enable_physics_purification=False, site_filter=None):
        """
        主执行流程
        data_path: 数据文件路径
        targets: 预测目标列表
        enable_physics_purification: 是否启用物理净化
        site_filter: 指定站点名称（仅分析该站点），None表示分析所有站点
        """
        # 保存数据文件名（用于输出文件夹命名）
        self.data_filename = os.path.basename(data_path)
        
        # 创建临时日志文件用于初始加载
        temp_log = os.path.join(self.root_dir, "加载日志.txt")
        
        # 加载数据
        df, has_sitename = self.load_data(data_path, temp_log)
        
        all_summaries = []
        
        if has_sitename:
            self.log(f"检测到 SITENAME 字段，启用站点分组分析模式", temp_log)
            
            # 获取所有站点
            sites = df['SITENAME'].unique()
            self.log(f"发现 {len(sites)} 个站点: {', '.join(sites)}", temp_log)
            
            if site_filter:
                # 检查指定站点是否存在
                if site_filter not in sites:
                    self.log(f"\n{'='*60}", temp_log)
                    self.log(f"提示: 指定的站点 '{site_filter}' 在数据中不存在", temp_log)
                    self.log(f"可用站点: {', '.join(sites)}", temp_log)
                    self.log(f"{'='*60}\n", temp_log)
                    print(f"\n提示: 指定的站点 '{site_filter}' 在数据中不存在")
                    print(f"可用站点: {', '.join(sites)}\n")
                    return
                else:
                    sites = [site_filter]
                    self.log(f"仅分析指定站点: {site_filter}", temp_log)
            
            # 按站点分组分析
            for site in sites:
                site_data = df[df['SITENAME'] == site].copy()
                summary = self.analyze_single_dataset(
                    site_data, site, targets, enable_physics_purification, self.root_dir
                )
                all_summaries.extend(summary)
        else:
            self.log(f"未检测到 SITENAME 字段，使用全局分析模式", temp_log)
            
            if site_filter:
                self.log(f"提示: 数据中不存在 SITENAME 字段，忽略站点过滤参数", temp_log)
            
            # 全局分析
            summary = self.analyze_single_dataset(
                df, None, targets, enable_physics_purification, self.root_dir
            )
            all_summaries.extend(summary)
            
            # 释放内存
            del df
            gc.collect()
        
        # 保存总汇总
        if all_summaries:
            final_summary_df = pd.DataFrame(all_summaries)
            final_summary_path = os.path.join(self.root_dir, "总体结果汇总.csv")
            final_summary_df.to_csv(final_summary_path, index=False, encoding='utf-8-sig')
            print(f"\n{'='*60}")
            print(f"所有分析已完成！")
            print(f"总体结果汇总: {final_summary_path}")
            print(f"{'='*60}\n")

    def save_pred_data(self, times, y_true, y_pred, target, model, paths):
        """保存预测数据"""
        df = pd.DataFrame({
            '时间': times if times is not None else range(len(y_true)),
            '真实值': y_true,
            '预测值': y_pred,
            '误差': y_true - y_pred
        })
        df.to_csv(os.path.join(paths['data'], f"{target}_{model}_预测详情.csv"),
                 index=False, encoding='utf-8-sig')

    def plot_results(self, times, y_true, y_pred, target, model, r2, rmse, mae, paths):
        """
        绘制图表（优化版）
        - 标签国际化（Observed/Predicted）
        - 简化标题
        - 添加指标框（R²、RMSE、MAE）
        - 添加拟合线信息
        """
        from scipy import stats
        
        # 格式化污染物名称
        formatted_target = format_pollutant_name(target)
        
        # 计算拟合线参数
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
        
        # 1. 时序图
        fig, ax = plt.subplots(figsize=(10, 4))
        if times is not None:
            ax.plot(times, y_true, 'k-', alpha=0.6, label='Observed', linewidth=1.5)
            ax.plot(times, y_pred, 'r--', alpha=0.8, label='Predicted', linewidth=1.5)
            plt.gcf().autofmt_xdate()
        else:
            ax.plot(y_true, 'k-', alpha=0.6, label='Observed', linewidth=1.5)
            ax.plot(y_pred, 'r--', alpha=0.8, label='Predicted', linewidth=1.5)
        
        ax.set_title(f"{formatted_target}-{model}", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(paths['series'], f"{target}_{model}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        
        # 2. 核密度估计散点图 (KDE Density Scatter Plot) - 用于论文
        if len(y_true) >= 50:
            try:
                from scipy.stats import gaussian_kde
                
                # 计算密度
                xy = np.vstack([y_true, y_pred])
                z = gaussian_kde(xy)(xy)
                
                # 按密度排序，让高密度点显示在上层
                idx = z.argsort()
                y_true_sorted, y_pred_sorted, z_sorted = y_true[idx], y_pred[idx], z[idx]
                
                fig, ax = plt.subplots(figsize=(7, 6))
                
                # 绘制密度散点
                scatter = ax.scatter(y_true_sorted, y_pred_sorted, c=z_sorted, 
                                    s=20, cmap='viridis', alpha=0.6, edgecolors='none')
                cbar = plt.colorbar(scatter, ax=ax, label='Kernel Density')
                
                # 绘制1:1线（理想拟合线）
                vmin = min(y_true.min(), y_pred.min())
                vmax = max(y_true.max(), y_pred.max())
                ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1.5, alpha=0.7, label='Ideal fit line')
                
                # 绘制拟合线（不显示label）
                fit_line = slope * np.array([vmin, vmax]) + intercept
                ax.plot([vmin, vmax], fit_line, 'r-', lw=2, alpha=0.8)
                
                # 设置标签和标题（使用统一格式的单位）
                ax.set_xlabel(f"Observed {formatted_target} ($\\mathrm{{\\mu g/m^3}}$)", fontsize=12)
                ax.set_ylabel(f"Predicted {formatted_target} ($\\mathrm{{\\mu g/m^3}}$)", fontsize=12)
                ax.set_title(f"{formatted_target}-{model}", fontsize=14, fontweight='bold')
                
                # 添加图例（放在左上角）
                ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
                
                # 添加指标文本（使用统一格式的单位）
                metrics_text = (
                    f'$R^2$ = {r2:.3f}\n'
                    f'MAE = {mae:.2f} $\\mathrm{{\\mu g/m^3}}$\n'
                    f'RMSE = {rmse:.2f} $\\mathrm{{\\mu g/m^3}}$'
                )
                # 添加指标文本（无边框，紧贴图例下方）
                ax.text(0.02, 0.91, metrics_text, transform=ax.transAxes,
                        fontsize=9, verticalalignment='top',
                        horizontalalignment='left', family='sans-serif',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                 edgecolor='none', alpha=0.85))
                
                ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
                ax.set_aspect('equal', adjustable='box')
                
                plt.tight_layout()
                plt.savefig(os.path.join(paths['scatter'], f"{target}_{model}_KDE.png"), dpi=300, bbox_inches='tight')
                plt.close('all')  # 释放内存
            except Exception as e:
                # 如果KDE计算失败，跳过
                plt.close('all')  # 确保失败时也关闭图表
                pass

    def run_shap(self, model, X, features, target, model_name, paths):
        """
        SHAP分析（优化版）
        - 限制显示Top 15特征
        - 优化图表样式
        - 英文标签
        """
        try:
            # 采样数据（避免计算量过大）
            X_sub = X[np.random.choice(X.shape[0], min(200, X.shape[0]), replace=False)]
            
            # 选择合适的explainer
            if model_name == 'linear_regression':
                explainer = shap.LinearExplainer(model, X_sub)
            else:
                explainer = shap.TreeExplainer(model)
            
            shap_values = explainer.shap_values(X_sub)
            
            # 创建SHAP summary plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 使用summary_plot，限制显示特征数量
            shap.summary_plot(
                shap_values, 
                X_sub, 
                feature_names=features,
                max_display=15,  # 只显示Top 15特征
                show=False,
                plot_size=(10, 6)
            )
            
            # 优化标题
            formatted_target = format_pollutant_name(target)
            plt.title(f"Feature Importance: {formatted_target}-{model_name}", 
                     fontsize=14, fontweight='bold', pad=15)
            
            # 优化xlabel
            plt.xlabel('SHAP Value (impact on model output)', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(paths['shap'], f"{target}_{model_name}_SHAP.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close('all')  # 释放内存
            
            # 显式删除临时变量
            del X_sub, explainer, shap_values
            gc.collect()  # 强制回收内存
            
        except Exception as e:
            # 如果SHAP分析失败，记录但不中断流程
            plt.close('all')  # 确保失败时也关闭图表
            print(f"SHAP analysis failed for {target}-{model_name}: {str(e)}")
            pass

    def calc_importance(self, model, features, target, model_name, paths):
        """计算特征权重"""
        imp = None
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
        elif hasattr(model, 'coef_'):
            imp = np.abs(model.coef_)
        if imp is not None:
            df = pd.DataFrame({'特征': features, '重要性': imp})
            df['占比(%)'] = (df['重要性'] / df['重要性'].sum() * 100).round(2)
            df.sort_values('占比(%)', ascending=False).to_csv(
                os.path.join(paths['data'], f"{target}_{model_name}_特征权重.csv"),
                index=False, encoding='utf-8-sig'
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='站点分组大气污染物回归分析')
    parser.add_argument('--data', type=str, required=True, help='数据文件路径（支持CSV/Excel）')
    parser.add_argument('--enable-physics-purification', action='store_true', default=False,
                       help='是否开启物理净化（防止数据泄露）')
    parser.add_argument('--site', type=str, default=None,
                       help='指定分析特定站点（仅当数据包含SITENAME字段时有效）')
    parser.add_argument('--targets', type=str, default='PM2.5,PM10,O3,SO2,NO2,CO',
                       help='预测目标，用逗号分隔（默认: PM2.5,PM10,O3,SO2,NO2,CO）')
    parser.add_argument('--output-dir', type=str, default='./分析结果',
                       help='输出目录（默认: ./分析结果）')
    
    args = parser.parse_args()
    
    # 解析targets
    targets = [t.strip() for t in args.targets.split(',')]
    
    # 运行分析
    pipeline = SiteBasedAnalysisPipeline(output_dir=args.output_dir)
    pipeline.run(
        data_path=args.data,
        targets=targets,
        enable_physics_purification=args.enable_physics_purification,
        site_filter=args.site
    )
