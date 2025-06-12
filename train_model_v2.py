import pandas as pd
import re
import numpy as np
import json
import joblib
import sys
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

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
    
    # Normalização avançada
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)  # Remover comentários /* */
    text = re.sub(r'(--|#).*?$', '', text, flags=re.MULTILINE)  # Remover comentários SQL
    text = re.sub(r';{2,}', ';', text)  # Normalizar múltiplos ;
    return text

# Função para tokenizar os statements SQL
def sql_tokenizer(text, patterns_json_path="sql_patterns.json"):
    with open(patterns_json_path, "r") as f:
        data = json.load(f)

    tokens = []
    for pattern in data["patterns"]:
        matches = re.finditer(pattern["regex"], text)
        for match in matches:
            tokens.append(match.group())  # Apenas a string capturada
    return tokens


# Função principal para treinar o modelo
def train_model(csv_file, model_output):
    data = pd.read_csv(csv_file).dropna()   # Carrega o CSV e remove NaNs
    if len(data) == 0:
        raise ValueError("Nenhum dado válido encontrado após remoção de NaNs")
    
    X = data.iloc[:, 0].astype(str)
    y = data.iloc[:, 1]
    
    # Calcular pesos de classe para lidar com desbalanceamento entre 0s e 1s
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))
    
    print(f"Distribuição de classes: {dict(pd.Series(y).value_counts())}")
    print(f"Pesos de classe: {class_weights}")
    
    # Configurar pipeline com SGDClassifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=sql_tokenizer,
            lowercase=True,
            max_features=5000,
            ngram_range=(1, 3),
            token_pattern=None
        )),
        ('clf', SGDClassifier(
            loss='log_loss',    # Regressão Logística
            penalty='l2',
            class_weight=class_weights,
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Treinamento com validação cruzada
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_f1 = 0
    best_model = None
    
    print("\nIniciando treinamento com validação cruzada...")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        print(f"Fold {fold+1} - F1-score: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = pipeline
            print(f"Novo melhor modelo no fold {fold+1}")
    
    # Treinar o melhor modelo com todos os dados
    best_model.fit(X, y)
    joblib.dump(best_model, model_output)
    
    print(f"\n✅ Modelo treinado e salvo em {model_output}")
    print(f"🎯 Melhor F1-score: {best_f1:.4f}")
    print(f"📊 Total de exemplos: {len(X)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python train_model.py <arquivo_csv> <modelo_saida.pkl>")
        sys.exit(1)
    
    train_model(sys.argv[1], sys.argv[2])