import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import optuna

import lightgbm as lgb
import catboost as cb
import xgboost as xgb

from sklearn.ensemble import StackingClassifier, RandomForestClassifier

import shap
import warnings
warnings.filterwarnings('ignore')

# Opcional: para deep learning tabular y AutoML
# from pytorch_tabnet.tab_model import TabNetClassifier
# import autogluon.tabular as ag
