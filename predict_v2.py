import joblib
import sys
import re
import json
import urllib.parse

# Dicionário para salvar modelos já carregados
MODEL_CACHE = {}

# Carregar configuração de padrões vindas de sql_patterns.json
with open('sql_patterns.json') as f:
    PATTERNS_CONFIG = json.load(f)

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


# Função para carregar modelo usando cache para evitar leituras repetidas
def load_cached_model(model_file):
    if model_file not in MODEL_CACHE:
        MODEL_CACHE[model_file] = joblib.load(model_file)
    return MODEL_CACHE[model_file]

# Detectar padrões de alto risco rapidamente ('OR 1=1', 'UNION SELECT', etc.)
def detect_high_risk_patterns(text):
    """Camada de regras rápidas para detecção imediata"""
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

# Função principal para predição
def predict_input(model_file, input_text):

    # Verificar dados binários
    if any(ord(char) < 32 and char not in '\t\n\r' for char in input_text):
        return 1, [0.0, 1.0], ["Dados binários detectados"]
    
    # Verificar pelas regras rápidas
    if detect_high_risk_patterns(input_text):
        return 1, [0.0, 1.0], ["Padrão crítico detectado"]
    
    # Carregar modelo
    pipeline = load_cached_model(model_file)
    
    # Tokenização
    try:
        vectorizer = pipeline.named_steps['tfidf']
        tokens = [t.lower() for t in sql_tokenizer(input_text)]  # Padroniza tokens para minúsculas
        vocab = set(vectorizer.get_feature_names_out())
        relevant_tokens = [t for t in tokens if t in vocab]
    except Exception as e:
        relevant_tokens = []
        print(f"Aviso: Tokenização falhou - {str(e)}", file=sys.stderr)
    
    # Predição
    prediction = pipeline.predict([input_text])[0]
    probability = pipeline.predict_proba([input_text])[0]
    
    return prediction, probability, relevant_tokens

# Soma o tokens detectado de acordo com os valores de peso em sql_patterns.json 
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
    
    # Baseado na probabilidade
    if prob[1] > 0.85:
        threat_level = "CRÍTICO"
        reasons.append("Probabilidade > 85%")
    elif prob[1] > 0.7:
        threat_level = "ALTO"
        reasons.append("Probabilidade > 70%")
    elif prob[1] > 0.5:
        threat_level = "MODERADO"
        reasons.append("Probabilidade > 50%")
    
    # Baseado na pontuação de tokens
    threat_score = calculate_threat_score(tokens)
    if threat_score > 3.0:
        threat_level = "CRÍTICO"
        reasons.append(f"Pontuação alta: {threat_score:.2f}")
    elif threat_score > 2.0:
        threat_level = "ALTO" if threat_level != "CRÍTICO" else threat_level
        reasons.append(f"Pontuação moderada: {threat_score:.2f}")
    
    # Detecção de combinações perigosas
    upper_tokens = [t.upper() for t in tokens]
    if 'UNION' in upper_tokens and 'SELECT' in upper_tokens:                # Immediate Banhammer!  
        threat_level = "ALTO" if threat_level == "BAIXO" else threat_level
        reasons.append("Combinação perigosa: UNION + SELECT")
    
    return threat_level, reasons

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python predict.py <modelo.pkl> <texto>")
        print("Opções:")
        print("  -f <arquivo>    Ler entrada de arquivo")
        print("  --json          Saída em formato JSON")
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
            print(f"Erro ao ler arquivo: {str(e)}", file=sys.stderr)
            sys.exit(1)
    else:
        # Remover flags e juntar argumentos
        args = [arg for arg in sys.argv[2:] if arg not in ["--json", "-f"]]
        input_text = " ".join(args)
    
    # Fazer predição
    try:
        pred, prob, tokens = predict_input(model_file, input_text)
        threat_level, reasons = analyze_threat_level(prob, tokens)
    except Exception as e:
        print(f"Erro na predição: {str(e)}", file=sys.stderr)
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
        print("\n" + "="*50)
        print("🔒 Análise de Segurança SQL Injection")
        print("="*50)
        print(f"\n📝 Entrada: {input_text[:250]}{'...' if len(input_text) > 250 else ''}")
        print(f"\n🔍 Tokens: {', '.join(set(tokens)) if tokens else 'Nenhum token relevante'}")
        print(f"\n📊 Resultado: {'🚨 INJEÇÃO DETECTADA' if pred == 1 else '✅ SEGURO'}")
        print(f"📈 Probabilidade: {prob[1]:.4f}")
        print(f"⚠️  Nível de Ameaça: {threat_level}")
        
        if reasons:
            print("\n🔎 Motivos:")
            for reason in reasons:
                print(f" - {reason}")
        
        print("\n" + "="*50)