# 한글과 영어 섞여있는 문장 거르기
## 방법1
'''
영어 단어만 추출하는 것. 반점 느낌표 등 다 무시
단점 : 키워드만 추출하기 때문에 - 와 같은 중요한 붙임표를 무시하게됌
'''

def extract_english(text):
    # 영어 단어만 추출
    english_parts = re.findall(r'\b[a-zA-Z]+\b', text)
    # 추출된 단어들을 공백으로 연결
    return ' '.join(english_parts)


print("원래 출력 : ", df_eng['Text'][0])
print("추출 후 출력 : ", extract_english(df_eng['Text'][0]))

## 방법2
'''
설치 오류
'''
# from polyglot.text import Text
# text = Text("Hello, 세상! Python programming is fun こんにちは、世界！")
# for word in text.words:
#     if word.language.code == 'en':
#         print(word)

## 방법3
'''
이것도 워드 토큰만 출력
'''
from nltk.tokenize import word_tokenize
from langdetect import detect

def extract_english_words(text):
    words = word_tokenize(text)
    english_words = [word for word in words if detect(word) == 'en']
    return ' '.join(english_words)

print("원래 출력 : ", df_eng['Text'][19899])
print("추출 후 출력 : ", extract_english(df_eng['Text'][19899]))

# 방법4 : fasttext
import fasttext
model_path = r'/.../lid.176.bin' # download : https://github.com/facebookresearch/fastText/?tab=readme-ov-file
model = fasttext.load_model(model_path)


non_english_texts = []  # 영어가 아닌 텍스트를 저장할 리스트
failed_detection_texts = []  # 언어 감지에 실패한 텍스트를 저장할 리스트

def fasttext_extract_eng(text):
    # 텍스트를 문장으로 분리 (간단한 예시로 '.'를 기준으로 분리)
    sentences = text.split('.')
    english_sentences = []

    for sentence in sentences:
        try:
            # fastText를 사용하여 각 문장의 언어를 감지
            pred = model.predict(sentence.strip())  # 공백 제거를 추가로 포함
            # 감지된 언어 코드가 영어('en')인 경우 리스트에 추가
            if pred[0][0] == '__label__en':
                english_sentences.append(sentence.strip())
                return True
            else:
                # 영어가 아닌 텍스트를 리스트에 추가
                non_english_texts.append(sentence.strip())
                return False
        except Exception as e:
            # 감지 실패한 텍스트를 리스트에 추가
            failed_detection_texts.append(sentence.strip())
            print(f"Detection failed for sentence: {sentence.strip()} with error {str(e)}")
            return False
