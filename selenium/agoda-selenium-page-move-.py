
import numpy as np
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.common.action_chains import ActionChains # 스크롤 액션

chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
chrome_options.add_argument("--incognito")

columns = ['Score', 'Country', 'Traveler Type', 'Room Type', 'Stay Duration', 'Title', 'Text', 'Date']
df = pd.DataFrame(columns = columns)

driver = webdriver.Chrome(options=chrome_options)

url = 'https://www.agoda.com/hostelling-international-new-york/hotel/new-york-ny-us.html?finalPriceView=1&isShowMobileAppPrice=false&cid=-1&numberOfBedrooms=&familyMode=false&adults=2&children=0&rooms=1&maxRooms=0&checkIn=2023-12-29&isCalendarCallout=false&childAges=&numberOfGuest=0&missingChildAges=false&travellerType=1&showReviewSubmissionEntry=false&currencyCode=USD&isFreeOccSearch=false&isCityHaveAsq=false&tspTypes=8&los=1&searchrequestid=88a67f2e-7b06-4af2-a372-15a58c592b9b&ds=zsdhNdtEGRbzqH3t'

driver.get(url)

next_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".ficon-carrouselarrow-right"))) # 이거 리뷰 넘기는거임

cnt = 1

target_numbs = 100

try:
    while True:
        # 현재 페이지의 리뷰 데이터 수집
        review_elements = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "Review-comment"))
        )

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

        cnt += 1
        print(cnt)

        if cnt == target_numbs :
            break

        
        action = ActionChains(driver)
        action.move_to_element(next_button).click().perform()

except Exception as e:
    print("Error:", e)

finally:
    driver.quit()

# 데이터프레임 저장
df.to_csv("example.csv", encoding = 'cp949')

pd.DataFrame.append