# 문자열 끝에서 " 문자를 제거하는 사용자 정의 함수

def remove_trailing_quote(text):
    if isinstance(text, str):  # 문자열인 경우에만 처리
        return text.rstrip('"')
    return text  # 문자열이 아닌 경우는 그대로 반환

