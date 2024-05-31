import re

def sub_special_characters(text):
    pattern = r'[^a-zA-Z가-힣0-9\s]'  # 알파벳, 숫자, 공백 문자가 아닌 모든 문자
    without_special_chars = re.sub(pattern, ' ', text)
    #without_special_chars = without_special_chars.lower() # 소문자로 내보내고 싶다면
    return ' '.join(without_special_chars.split()).strip()