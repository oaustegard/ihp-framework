import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from scipy.stats import zscore
from textblob import TextBlob
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataProcessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.numeric_imputer = KNNImputer(n_neighbors=5)
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.text_tokenizer = Tokenizer()

    def load_data(self, file_path, data_type):
        if data_type == 'csv':
            return pd.read_csv(file_path)
        elif data_type == 'json':
            return pd.read_json(file_path)
        elif data_type == 'excel':
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    def preprocess_data(self, data, numeric_columns=None, categorical_columns=None, text_columns=None):
        # Handle missing values
        if numeric_columns:
            data[numeric_columns] = self.numeric_imputer.fit_transform(data[numeric_columns])
        if categorical_columns:
            data[categorical_columns] = self.categorical_imputer.fit_transform(data[categorical_columns])

        # Scale numeric columns
        if numeric_columns:
            for col in numeric_columns:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                data[col] = self.scalers[col].fit_transform(data[[col]])

        # Encode categorical columns
        if categorical_columns:
            for col in categorical_columns:
                if col not in self.encoders:
                    self.encoders[col] = OneHotEncoder(sparse=False, handle_unknown='ignore')
                encoded = self.encoders[col].fit_transform(data[[col]])
                encoded_df = pd.DataFrame(encoded, columns=[f"{col}_{cat}" for cat in self.encoders[col].categories_[0]])
                data = pd.concat([data, encoded_df], axis=1)
                data.drop(col, axis=1, inplace=True)

        # Process text columns
        if text_columns:
            for col in text_columns:
                data[f"{col}_processed"] = data[col].apply(self.preprocess_text)
                data[f"{col}_sentiment"] = data[col].apply(self.get_sentiment)
            self.text_tokenizer.fit_on_texts(data[text_columns].values.flatten())
            for col in text_columns:
                sequences = self.text_tokenizer.texts_to_sequences(data[f"{col}_processed"])
                data[f"{col}_encoded"] = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as needed

        return data

    def preprocess_text(self, text):
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        return text

    def get_sentiment(self, text):
        return TextBlob(text).sentiment.polarity

    def integrate_data_sources(self, data_sources):
        integrated_data = pd.DataFrame()
        for source, data in data_sources.items():
            if integrated_data.empty:
                integrated_data = data
            else:
                common_columns = integrated_data.columns.intersection(data.columns)
                integrated_data = pd.merge(integrated_data, data[common_columns], 
                                           how='outer', left_index=True, right_index=True, 
                                           suffixes=(f'_{source}', ''))
        
        # Handle conflicts in merged data
        for col in integrated_data.columns:
            if col.endswith('_x') or col.endswith('_y'):
                base_col = col[:-2]
                integrated_data[base_col] = integrated_data[base_col].combine_first(integrated_data[col])
                integrated_data.drop(col, axis=1, inplace=True)
        
        return integrated_data

    def create_time_series_dataset(self, data, lookback, forecast_horizon):
        X, y = [], []
        for i in range(len(data) - lookback - forecast_horizon + 1):
            X.append(data[i:(i + lookback)])
            y.append(data[i + lookback:i + lookback + forecast_horizon])
        return np.array(X), np.array(y)

    def detect_anomalies(self, data, threshold=3):
        return np.abs(zscore(data)) > threshold

    def perform_feature_selection(self, X, y, k=10):
        from sklearn.feature_selection import SelectKBest, f_regression
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        return X_selected, selected_features

class TemporalGraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()

    def build_graph(self, events, time_window):
        for i, event in enumerate(events):
            self.graph.add_node(i, **event)
        
        for i in range(len(events)):
            for j in range(i+1, len(events)):
                if abs(events[i]['timestamp'] - events[j]['timestamp']) <= time_window:
                    self.graph.add_edge(i, j)
        
        return self.graph

    def extract_temporal_motifs(self, motif_size):
        motifs = list(nx.enumerate_all_cliques(self.graph, min_size=motif_size, max_size=motif_size))
        return motifs

    def compute_centrality(self):
        return nx.eigenvector_centrality(self.graph)

    def detect_communities(self):
        return nx.community.louvain_communities(self.graph)

    def compute_temporal_pagerank(self, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6):
        return nx.pagerank(self.graph, alpha=alpha, personalization=personalization, max_iter=max_iter, tol=tol)

class TimeSeriesAnalyzer:
    def __init__(self):
        pass

    def decompose_time_series(self, data, period=None):
        from statsmodels.tsa.seasonal import seasonal_decompose
        result = seasonal_decompose(data, period=period, model='additive')
        return result.trend, result.seasonal, result.resid

    def perform_granger_causality(self, data1, data2, max_lag=5):
        from statsmodels.tsa.stattools import grangercausalitytests
        data = pd.concat([data1, data2], axis=1)
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        return results

    def compute_cross_correlation(self, data1, data2, max_lag=None):
        return np.correlate(data1, data2, mode='full')

    def detect_change_points(self, data, penalty=None):
        from ruptures import Binseg
        model = Binseg(model="l2").fit(data)
        change_points = model.predict(pen=penalty)
        return change_points

    def forecast_arima(self, data, order=(1,1,1), seasonal_order=(0,0,0,0), forecast_horizon=10):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
        results = model.fit()
        forecast = results.forecast(steps=forecast_horizon)
        return forecast
