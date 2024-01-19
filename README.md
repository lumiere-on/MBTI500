# 필요한 라이브러리 임포트
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

# 데이터 로드
data = pd.read_excel('MBTI 500.csv')  # 파일 경로에 맞게 수정

# 텍스트 데이터와 레이블 분리
X = data['text']
y = data['mbti']

# 레이블 인코딩
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 데이터를 훈련 세트와 테스트 세트로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 텍스트를 토큰화하고 패딩하는 함수
def tokenize_and_pad(texts, max_len=50):
    # 텍스트를 토큰화하고 패딩
    tokenized_texts = [text.split() for text in texts]
    padded_texts = [tokens[:max_len] + ['<pad>'] * max(0, max_len - len(tokens)) for tokens in tokenized_texts]
    return padded_texts

# 훈련 데이터와 테스트 데이터 토큰화 및 패딩
max_len = 50  # 문장의 최대 길이
X_train_tokens = tokenize_and_pad(X_train, max_len)
X_test_tokens = tokenize_and_pad(X_test, max_len)

# 단어를 정수로 매핑하는 사전 만들기
word_to_idx = {'<pad>': 0}
for tokens in X_train_tokens + X_test_tokens:
    for token in tokens:
        if token not in word_to_idx:
            word_to_idx[token] = len(word_to_idx)

# 정수를 단어로 매핑하는 사전 만들기
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# 토큰을 정수로 변환하는 함수
def tokens_to_indices(tokens):
    return [word_to_idx[token] for token in tokens]

# 정수 시퀀스로 변환
X_train_indices = [tokens_to_indices(tokens) for tokens in X_train_tokens]
X_test_indices = [tokens_to_indices(tokens) for tokens in X_test_tokens]

# PyTorch DataLoader를 사용하여 데이터 로딩
batch_size = 64
train_dataset = TensorDataset(torch.tensor(X_train_indices), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(torch.tensor(X_test_indices), torch.tensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1, :, :])
        output = self.softmax(output)
        return output

# 모델 초기화
vocab_size = len(word_to_idx)
embedding_dim = 50
hidden_dim = 100
output_dim = len(label_encoder.classes_)
lstm_model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim)

# 손실 함수 및 최적화 알고리즘 정의
criterion = nn.CrossEntropyLoss()
optimizer = Adam(lstm_model.parameters(), lr=0.001)

# 훈련 함수 정의
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# 훈련 진행
train_model(lstm_model, train_loader, criterion, optimizer)

# 테스트 데이터에 대한 예측 및 평가
lstm_model.eval()
all_preds = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = lstm_model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())

# 정확도 및 리포트 출력
y_test_encoded = label_encoder.transform(y_test)
accuracy = accuracy_score(y_test_encoded, all_preds)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test_encoded, all_preds))
