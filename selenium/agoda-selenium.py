
import numpy as np
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 브라우저 꺼짐 방지
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)

# 빈 데이터프레임 생성
columns = ['Score', 'Country', 'Traveler Type', 'Room Type', 'Stay Duration', 'Title', 'Text', 'Date']
df = pd.DataFrame(columns=columns)

# 웹 드라이버 설정
browser = webdriver.Chrome(options=chrome_options)

# 웹사이트 열기
url = 'https://www.agoda.com/ko-kr/hilton-times-square-hotel/hotel/new-york-ny-us.html?finalPriceView=1&isShowMobileAppPrice=false&cid=1891463&numberOfBedrooms=&familyMode=false&adults=2&children=0&rooms=1&maxRooms=0&checkIn=2023-12-14&isCalendarCallout=false&childAges=&numberOfGuest=0&missingChildAges=false&travellerType=1&showReviewSubmissionEntry=false&currencyCode=KRW&isFreeOccSearch=false&tag=45b17d1d-e0b0-fe2a-ce90-5513829d856b&isCityHaveAsq=false&los=1&searchrequestid=48d4bfa9-c94f-4230-b293-31bdcf6e97c7&ds=EQDe0O9a%2FbZzOyX5'

browser.get(url)

try:
    # "Review-comment" 클래스를 가진 모든 요소를 찾음
    review_elements = WebDriverWait(browser, 10).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, "Review-comment"))
    )

    for review_element in review_elements:
        # 리뷰 세부 정보 추출
        score = review_element.find_element(By.CLASS_NAME, "Review-comment-leftScore").text
        country = review_element.find_element(By.CLASS_NAME, "flag").get_attribute("class").split("-")[-1].upper()
        traveler_type = review_element.find_elements(By.CLASS_NAME, "Review-comment-reviewer")[1].text
        room_type = review_element.find_elements(By.CLASS_NAME, "Review-comment-reviewer")[2].text
        stay_duration = review_element.find_elements(By.CLASS_NAME, "Review-comment-reviewer")[3].text
        title = review_element.find_element(By.CLASS_NAME, "Review-comment-bodyTitle").text
        text = review_element.find_element(By.CLASS_NAME, "Review-comment-bodyText").text
        date = review_element.find_element(By.CLASS_NAME, "Review-statusBar-date").text

        # 데이터프레임에 행 추가
        df = df.append({
            'Score': score, 
            'Country': country, 
            'Traveler Type': traveler_type, 
            'Room Type': room_type, 
            'Stay Duration': stay_duration, 
            'Title': title, 
            'Text': text, 
            'Date': date
        }, ignore_index=True)

except Exception as e:
    print("Error:", e)

finally:
    browser.quit()

# 데이터프레임 출력
print(df)
# df.to_csv("example.csv", encoding = 'cp949')

