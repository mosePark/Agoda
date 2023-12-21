
import numpy as np
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
chrome_options.add_argument("--incognito")

columns = ['Score', 'Country', 'Traveler Type', 'Room Type', 'Stay Duration', 'Title', 'Text', 'Date']
df = pd.DataFrame(columns=columns)

driver = webdriver.Chrome(options=chrome_options)

url = 'https://www.agoda.com/ko-kr/hilton-times-square-hotel/hotel/new-york-ny-us.html?finalPriceView=1&isShowMobileAppPrice=false&cid=1891463&numberOfBedrooms=&familyMode=false&adults=2&children=0&rooms=1&maxRooms=0&checkIn=2023-12-14&isCalendarCallout=false&childAges=&numberOfGuest=0&missingChildAges=false&travellerType=1&showReviewSubmissionEntry=false&currencyCode=KRW&isFreeOccSearch=false&tag=45b17d1d-e0b0-fe2a-ce90-5513829d856b&isCityHaveAsq=false&los=1&searchrequestid=48d4bfa9-c94f-4230-b293-31bdcf6e97c7&ds=EQDe0O9a%2FbZzOyX5'

driver.get(url)

target_page = 1

try:
    while True:
        

        current_page_element = driver.find_element(By.CLASS_NAME, 'Review-paginator-number--current')
        current_page_number = int(current_page_element.text)

        print(f"현재 페이지 {current_page_number}에 도달했습니다.")

        target_page_element = driver.find_element(By.XPATH, f"//span[@class='Typographystyled__TypographyStyled-sc-j18mtu-0 cfuHGb kite-js-Typography Review-paginator-number ' and text()='{target_page}']")
        target_page_element.click()
        target_page += 1

        time.sleep(2)  # 페이지 로딩 대기

        review_elements = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "Review-comment"))
        )

        if target_page == 100 :
            break

        for review_element in review_elements:
            score = review_element.find_element(By.CLASS_NAME, "Review-comment-leftScore").text
            country = review_element.find_element(By.CLASS_NAME, "flag").get_attribute("class").split("-")[-1].upper()
            traveler_type = review_element.find_elements(By.CLASS_NAME, "Review-comment-reviewer")[1].text
            room_type = review_element.find_elements(By.CLASS_NAME, "Review-comment-reviewer")[2].text
            stay_duration = review_element.find_elements(By.CLASS_NAME, "Review-comment-reviewer")[3].text
            title = review_element.find_element(By.CLASS_NAME, "Review-comment-bodyTitle").text
            text = review_element.find_element(By.CLASS_NAME, "Review-comment-bodyText").text
            date = review_element.find_element(By.CLASS_NAME, "Review-statusBar-date").text

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
    driver.quit()

# 데이터프레임 저장
df.to_csv("example.csv", encoding = 'cp949')