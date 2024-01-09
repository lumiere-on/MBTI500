# MBTI500
Hopy Winter project
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# mbti별 빈도수가 낮은 단어들
mbti_low_frequency_words = {
    'infp': ['Quadra', 'critical', 'really', 'post', 'response'],
    'esfp': ['aura', 'supervisor', 'allegro', 'rigorous', 'hub'],
    'entp': ['secure', 'step', 'actual', 'implement', 'sometimes'],
    'isfp': ['always', 'regardless', 'still', 'chat', 'along'],
    'intj': ['tool', 'people', 'interaction', 'excuse', 'use'],
    'entj': ['well', 'right', 'psychoanal', 'point', 'mourn'],
    'intp': ['goal', 'personally', 'appeal', 'probably', 'wrong'],
    'istp': ['like', 'hard', 'friend', 'interest', 'college'],
    'esfj': ['relative', 'component', 'classical', 'thread', 'ambient'],
    'enfj': ['repeatedly', 'unconscious', 'recharge', 'superficial', 'inferior'],
    'enfp': ['judgement', 'unhealthy', 'allegation', 'acceptance', 'brainstorm'],
    'estp': ['stereotype', 'accreditation', 'accountability', 'realness', 'experience'],
    'estj': ['cognition', 'intention', 'gryffindor', 'stringent', 'introspect'],
    'infj': ['clarify', 'onus', 'endorse', 'coerce', 'disclaimer'],
    'isfj': ['subjective', 'disastrous', 'platonic', 'subconscious', 'enneagram'],
    'istj': ['aesthetic', 'roommate', 'mundane', 'catastrophe', 'procrastination']
}

# 예시 텍스트 데이터
texts = [
    "This is an example sentence with Quadra and really. It also contains aesthetic.",
    # 다른 예시 텍스트들...
]

# 각 텍스트에 대한 mbti 타입
mbti_types = ['infp', 'esfp', 'entp', 'isfp', 'intj', 'entj', 'intp', 'istp', 'esfj', 'enfj', 'enfp', 'estp', 'estj', 'infj', 'isfj', 'istj']

# 텍스트 전처리 함수
def preprocess_mbti_text(text, mbti_type):
    # 텍스트를 소문자로 변환
    text = text.lower()

    # 구두점 제거
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)

    # 불용어 제거
    stop_words = set(['the', 'and', 'is', 'in', 'it', 'to', 'of', 'this'])  # 간단한 예시
    words = [word for word in text.split() if word.lower() not in stop_words]

    # 어간 추출 (여기서는 단어를 그대로 반환)
    words = [word for word in words]

    # mbti별 빈도수가 낮은 단어 제거
    words = [word for word in words if word not in mbti_low_frequency_words.get(mbti_type, [])]

    return ' '.join(words)

# 전처리된 텍스트 생성
preprocessed_texts = [preprocess_mbti_text(text, mbti_type) for text, mbti_type in zip(texts, mbti_types)]

# CountVectorizer를 사용하여 텍스트를 숫자로 변환
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(preprocessed_texts)

# 데이터 분할 (예시: 80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, mbti_types, test_size=0.2, random_state=42)

# 결과 출력
print("전처리 후 텍스트 데이터:")
print(preprocessed_texts)
print("\n숫자로 변환된 데이터:")
print(X.toarray())
print("\n학습 데이터 크기:", X_train.shape)
print("테스트 데이터 크기:", X_test.shape)
