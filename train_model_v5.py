import pandas as pd
import re
import numpy as np
import json
import joblib
import copy
import sys
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import FunctionTransformer

# Carregar padrões de tokenização de sql_patterns.json
with open('sql_patterns.json') as f:
    PATTERNS_CONFIG = json.load(f)

SQL_PATTERNS = [p['regex'] for p in PATTERNS_CONFIG['patterns']]

# Função de normalização dos statements SQL
def normalize_text(text):
    text = str(text)
    text = urllib.parse.unquote(text)
    text = re.sub(r'%([0-9a-fA-F]{2})', lambda m: chr(int(m.group(1), 16)), text)
    text = re.sub(r'\\x([0-9a-fA-F]{2})', lambda m: chr(int(m.group(1), 16)), text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'(--|#).*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r';{2,}', ';', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Função para substituir a lambda
def normalize_series(series):
    return series.apply(normalize_text)

# Função para tokenizar os statements SQL
def sql_tokenizer(text):
    tokens = []
    for pattern in SQL_PATTERNS:
        matches = re.finditer(pattern, text)
        for match in matches:
            tokens.append(match.group())
    return tokens

# Função para análise da matriz de confusão
def analyze_confusion_matrix(cm):
    vp, fp, fn, vn = cm.ravel()
    
    metrics = {
        "Acurácia": (vp + vn) / (vp + fp + fn + vn),
        "Precisão": vp / (vp + fp) if (vp + fp) > 0 else 0,
        "Recall (Sensibilidade)": vp / (vp + fn) if (vp + fn) > 0 else 0,
        "Especificidade": vn / (vn + fp) if (vn + fp) > 0 else 0,
        "F1-Score": 2 * vp / (2 * vp + fp + fn) if (2 * vp + fp + fn) > 0 else 0
    }
    
    print("\n🧮 Métricas Calculadas:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

# Função principal para treinar o modelo
def train_model(csv_file, model_output):
    data = pd.read_csv(csv_file).dropna()
    if len(data) == 0:
        raise ValueError("Nenhum dado válido encontrado após remoção de NaNs")
    
    X = data.iloc[:, 0].astype(str)
    y = data.iloc[:, 1]
    
    # Calcular pesos de classe
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))
    
    print(f"Distribuição de classes: {dict(pd.Series(y).value_counts())}")
    print(f"Pesos de classe: {class_weights}")
    
    # Pipeline com função nomeada
    pipeline = Pipeline([
        ('normalize', FunctionTransformer(normalize_series)),  # Função nomeada
        ('tfidf', TfidfVectorizer(
            tokenizer=sql_tokenizer,
            lowercase=True,
            max_features=5000,
            ngram_range=(1, 3),
            token_pattern=None
        )),
        ('clf', SGDClassifier(
            loss='log_loss',
            penalty='l2',
            class_weight=class_weights,
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Treinamento com validação cruzada
    skf = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
    best_f1 = 0
    best_model = None
    all_y_test = []
    all_y_pred = []
    
    print("\nIniciando treinamento com validação cruzada...")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        
        # Matriz de confusão por fold
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nFold {fold+1} - F1-score: {f1:.4f}")
        print(f"Matriz de Confusão (Fold {fold+1}):")
        print(cm)
        
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = copy.deepcopy(pipeline)
            print(f"Novo melhor modelo no fold {fold+1}")
    
    # Matriz de confusão agregada
    cm_aggregated = confusion_matrix(all_y_test, all_y_pred)
    print("\nMatriz de Confusão Agregada:")
    print(cm_aggregated)
    
    # Métricas detalhadas
    vp, fp, fn, vn = cm_aggregated.ravel()
    print("\n📊 Métricas de Desempenho:")
    print(f"Verdadeiros Positivos (VP): {vp}")
    print(f"Falsos Positivos (FP): {fp}")
    print(f"Falsos Negativos (FN): {fn}")
    print(f"Verdadeiros Negativos (VN): {vn}")
    
    analyze_confusion_matrix(cm_aggregated)
    
    # Treinar o melhor modelo com todos os dados
    best_model.fit(X, y)
    joblib.dump(best_model, model_output)
    
    print(f"\n✅ Modelo treinado e salvo em {model_output}")
    print(f"🎯 Melhor F1-score: {best_f1:.4f}")
    print(f"📊 Total de exemplos: {len(X)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python train_model_v3.py <arquivo_csv> <modelo_saida.pkl>")
        sys.exit(1)
    
    train_model(sys.argv[1], sys.argv[2])