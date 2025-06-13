import joblib
import sys
import re
import json
import urllib.parse
import logging

# Configurar log
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Dicionário para cache de modelos
MODEL_CACHE = {}

# Carregar configuração de padrões
try:
    with open('sql_patterns.json') as f:
        PATTERNS_CONFIG = json.load(f)
except Exception as e:
    logging.error(f"Erro ao carregar sql_patterns.json: {e}")
    sys.exit(1)

SQL_PATTERNS = [p['regex'] for p in PATTERNS_CONFIG['patterns']]
TOKEN_WEIGHTS = {p['regex']: p.get('weight', 1.0) for p in PATTERNS_CONFIG['patterns']}

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

# Tokenização
def sql_tokenizer(text):
    tokens = []
    for pattern in SQL_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            tokens.append(match.group())
    return tokens


# Função para carregar modelo usando cache para evitar leituras repetidas
def load_cached_model(model_file):
    if model_file not in MODEL_CACHE:
        MODEL_CACHE[model_file] = joblib.load(model_file)
    return MODEL_CACHE[model_file]

# Detectar padrões de alto risco rapidamente ('OR 1=1', 'UNION SELECT', etc.)
def detect_high_risk_patterns(text):
    text = normalize_text(text).upper()
    high_risk_rules = [
        r'\bOR\s*1\s*=\s*1\b',
        r'\bUNION\s+SELECT\b',
        r';\s*(--|#)',
        r'\bDROP\s+TABLE\b',
        r'\bEXEC\s*\(.+\)',
        r'\bXP_CMDSHELL\b'
    ]
    return any(re.search(rule, text) for rule in high_risk_rules)

# Score de ameaça
def calculate_threat_score(tokens):
    score = 0.0
    for token in tokens:
        for pattern in PATTERNS_CONFIG['patterns']:
            if re.search(pattern['regex'], token, re.IGNORECASE):
                score += pattern.get('weight', 1.0)
                break
    return score

# Analisar o nível de ameaça baseado na probabilidade e tokens encontrados
def analyze_threat_level(prob, tokens):
    threat_level = "BAIXO"
    reasons = []

    # Análise por probabilidade
    if prob[1] > 0.85:
        threat_level = "CRÍTICO"
        reasons.append("Probabilidade > 85%")
    elif prob[1] > 0.7:
        threat_level = "ALTO"
        reasons.append("Probabilidade > 70%")
    elif prob[1] > 0.5:
        threat_level = "MODERADO"
        reasons.append("Probabilidade > 50%")

    # Análise por tokens
    threat_score = calculate_threat_score(tokens)
    if threat_score > 3.0:
        threat_level = "CRÍTICO"
        reasons.append(f"Pontuação alta: {threat_score:.2f}")
    elif threat_score > 2.0 and threat_level != "CRÍTICO":
        threat_level = "ALTO"
        reasons.append(f"Pontuação moderada: {threat_score:.2f}")
    
    # Detecção de padrão crítico
    upper_tokens = [t.upper() for t in tokens]
    if 'UNION' in upper_tokens and 'SELECT' in upper_tokens:
        if threat_level == "BAIXO":
            threat_level = "ALTO"
        reasons.append("Combinação perigosa: UNION + SELECT")

    return threat_level, reasons

# Predição principal
def predict_input(model_file, input_text):
    # Verificar por binários (exceto espaços em branco permitidos)
    if re.search(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', input_text):
        return 1, [0.0, 1.0], ["Dados binários detectados"]

    if detect_high_risk_patterns(input_text):
        return 1, [0.0, 1.0], ["Padrão crítico detectado"]

    pipeline = load_cached_model(model_file)

    # Normalizar a entrada antes da predição 
    normalized_input = normalize_text(input_text)

    try:
        tokens = [t.lower() for t in sql_tokenizer(normalized_input)]
        if 'tfidf' in pipeline.named_steps:
            vocab = set(pipeline.named_steps['tfidf'].get_feature_names_out())
            relevant_tokens = [t for t in tokens if t in vocab]
        else:
            relevant_tokens = tokens
    except Exception as e:
        logging.warning(f"Tokenização falhou: {e}")
        relevant_tokens = []

    try:
        prediction = pipeline.predict([normalized_input])[0]
        probability = pipeline.predict_proba([normalized_input])[0]
    except Exception as e:
        logging.error(f"Erro ao predizer: {e}")
        return 1, [0.0, 1.0], ["Erro ao predizer"]

    return prediction, probability, relevant_tokens

# Interface de linha de comando
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python predict_v2.py <modelo.pkl> <texto> [-f arquivo] [--json]")
        sys.exit(1)

    model_file = sys.argv[1]
    use_json = "--json" in sys.argv
    file_input = "-f" in sys.argv
    
    # Ler entrada
    input_text = ""
    if file_input:
        try:
            with open(sys.argv[sys.argv.index("-f") + 1], 'r', encoding='utf-8', errors='ignore') as f:
                input_text = f.read()
        except Exception as e:
            logging.error(f"Erro ao ler arquivo: {e}")
            sys.exit(1)
    else:
        args = [arg for arg in sys.argv[2:] if arg not in ["--json", "-f"]]
        input_text = " ".join(args)

    try:
        pred, prob, tokens = predict_input(model_file, input_text)
        threat_level, reasons = analyze_threat_level(prob, tokens)
    except Exception as e:
        logging.error(f"Erro na predição: {e}")
        sys.exit(1)
    
    # Gerar saída
    if use_json:
        output = {
            "prediction": int(pred),
            "probability": float(prob[1]),
            "threat_level": threat_level,
            "tokens": tokens,
            "reasons": reasons,
            "input_preview": input_text[:250] + ("..." if len(input_text) > 250 else "")
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print("\n" + "=" * 50)
        print("🔒 Análise de Segurança SQL Injection")
        print("=" * 50)
        print(f"\n📝 Entrada: {input_text[:250]}{'...' if len(input_text) > 250 else ''}")
        print(f"\n🔍 Tokens: {', '.join(set(tokens)) if tokens else 'Nenhum token relevante'}")
        print(f"\n📊 Resultado: {'🚨 INJEÇÃO DETECTADA' if pred == 1 else '✅ SEGURO'}")
        print(f"📈 Probabilidade: {prob[1]:.4f}")
        print(f"⚠️  Nível de Ameaça: {threat_level}")
        if reasons:
            print("\n🔎 Motivos:")
            for reason in reasons:
                print(f" - {reason}")
        print("\n" + "=" * 50)