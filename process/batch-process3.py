import numpy as np
import pandas as pd

import time
import sys
import os
import json

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import ElementClickInterceptedException

####################################################################################
####################################################################################
# 광고 닫기 버튼                                                                    
def close_ad_popup():
    try:
        ad_buttons = driver.find_elements(By.CLASS_NAME, "ab-message-button")
        for bt in ad_buttons:
            if bt.text == '닫기':
                bt.click()
                break
    except Exception:
        pass  # 광고가 없으면 무시      
####################################################################################
####################################################################################
# 광고2 닫기 버튼
def dismiss_popup():
    try:
        dismiss_button = driver.find_element(By.CLASS_NAME, "BackToSearch-dismissText")
        dismiss_button.click()
    except Exception :
        pass  # 버튼이 없거나 클릭할 수 없는 상태이면 무시
####################################################################################
####################################################################################
# 광고3 닫기 버튼
def close_no_thanks_popup():
    try:
        ad_buttons = driver.find_elements(By.CLASS_NAME, "ab-message-button")
        for bt in ad_buttons:
            if bt.text == 'No thanks':
                bt.click()
                break
    except NoSuchElementException:
        pass  # 버튼이 없으면 무시

def collect_reviews(driver, hotel_name):
    new_df = pd.DataFrame()
    try :
        review_elements = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, "Review-comment"))
        )

        for review_element in review_elements:
            reviewer_elem = review_element.find_elements(By.CLASS_NAME, "Review-comment-reviewer")
            score = review_element.find_element(By.CLASS_NAME, "Review-comment-leftScore").text
            country_elements = review_element.find_elements(By.CLASS_NAME, "flag")
            country = country_elements[0].get_attribute("class").split("-")[-1].upper() if country_elements else None

            review_page = review_page_cnt

            # 여행자 유형 추출
            try:
                group_name_elements = review_element.find_elements(By.XPATH, ".//div[@data-info-type='group-name']")
                traveler_type = group_name_elements[0].text

            except Exception :
                traveler_type = None

            # 룸 타입 추출
            try:
                room_type_elements = review_element.find_elements(By.XPATH, ".//div[@data-info-type='room-type']")
                room_type = room_type_elements[0].text

            except Exception :
                room_type = None

            # 숙박 기간 추출
            try:
                stay_detail_elements = review_element.find_elements(By.XPATH, ".//div[@data-info-type='stay-detail']")
                stay_duration = stay_detail_elements[0].text
            except Exception :
                stay_duration = None
            
            title = review_element.find_element(By.CLASS_NAME, "Review-comment-bodyTitle").text
            text = review_element.find_element(By.CLASS_NAME, "Review-comment-bodyText").text
            date = review_element.find_element(By.CLASS_NAME, "Review-statusBar-date").text

            new_review = {
                'hotel_name': hotel_name,
                'Score': score, 
                'Country': country, 
                'Traveler Type': traveler_type, 
                'Room Type': room_type, 
                'Stay Duration': stay_duration, 
                'Title': title, 
                'Text': text, 
                'Date': date,
                'review_page' : review_page
            }

            new_df = pd.concat([new_df, pd.DataFrame([new_review])], ignore_index=True)


    except Exception :
        print(f"현재 {nms}호텔의 {review_page_cnt}페이지 크롤링이 어렵습니다. 수집을 생략합니다.")
        pass

    return new_df
####################################################################################
####################################################################################
with open('C:/Users/UOS/proj_0/selenium/Code/all-hotel-web.txt', 'r') as file:
    dic = json.load(file)

hotel_nms = pd.read_csv("C:/Users/UOS/proj_0/selenium/Code/batch-process-3.csv", encoding = 'utf-8')

hotel_nms = hotel_nms['Hotel_Name'].tolist()
####################################################################################
####################################################################################

sys.stdout = open(f'C:/Users/UOS/proj_0/selenium/debug-log-3/debug-log.txt', 'w')
    
# 드라이버 마운트
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
chrome_options.add_argument("--incognito")

driver = webdriver.Chrome(options=chrome_options)

#호텔 빈 데이터셋 설정
cols = ['hotel_name','Score', 'Country', 'Traveler Type', 'Room Type', 'Stay Duration', 'Title', 'Text', 'Date', 'review_page']
df = pd.DataFrame(columns = cols)

# hotel_nms.remove("HOTEL MERIT")
# hotel_nms.remove("The Gallery House")

# hotel_nms[dic['The Gallery House']]

# hotel_nms = hotel_nms[109:]

# hotel_nms = hotel_nms[31:]

for nms in hotel_nms :
    url = dic[nms]

    driver.get(url)
    #
    close_ad_popup()
    close_no_thanks_popup()
    #

    driver.execute_script("window.scrollBy(0, 0);")

    time.sleep(3)
    # 언어 변경
    languge_button = driver.find_element(By.XPATH, "//*[@id='page-header']/section/div[2]/div[1]/div[2]/div[2]/div")
    languge_button.click()

    time.sleep(3)
    #
    close_ad_popup()
    close_no_thanks_popup()
    #
    # eng_button = driver.find_element(By.CSS_SELECTOR, "button[data-selenium='language-popup-menu-item-box']")
    # eng_button = driver.find_element(By.XPATH, "/html/body/div[22]/div/div[2]/div/div[2]/div[4]/div[2]")
    # eng_button = driver.find_element(By.XPATH, "/html/body/div[15]/div/div[2]/div/div[2]/div[4]/div[2]")
    # eng_button = driver.find_element(By.CLASS_NAME, "avuwS")
    eng_button = driver.find_element(By.XPATH, "//button[contains(., 'English')]")
    eng_button.click()

    


    time.sleep(3)
    #
    close_ad_popup()
    close_no_thanks_popup()
    #
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # ## 버튼 추천이 매일 달라짐 다른 카테고리에서 클릭하도록 !
    # 통화 변경 버튼 클릭
    price_button = driver.find_element(By.XPATH, "//*[@id='page-header']/section/div[2]/div[1]/div[2]/div[1]/div")
    price_button.click()

    # 첫 번째 방법 시도
    try:
        time.sleep(3)  # 필요한 경우 로딩 대기
        dol_button = driver.find_element(By.XPATH, "//button[contains(., '미국 달러')]")
        dol_button.click()
    except NoSuchElementException:
        # 첫 번째 방법 실패 시 두 번째 방법 시도
        action = ActionChains(driver)
        dol_button = driver.find_element(By.XPATH, "//button[contains(., 'US Dollar')]")
        action.move_to_element(dol_button).click().perform()

    #
    close_ad_popup()
    close_no_thanks_popup()
    #

    while True :
        try :
            dismiss_popup()
            agoda_reviews_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "div.Review-tabs > span[data-provider-id='332']")))
            agoda_reviews_button.click()
            break

        except Exception as e :
            print(f"{nms} 호텔 아고다리뷰버튼을 찾지 못했습니다.", e)
            break

    
    review_page_cnt = 1
    while True :
        try :
            next_bt = WebDriverWait(driver, 7).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "#reviewSection > div:nth-child(4) > div > span:nth-child(3)"))
            )
        except :
            next_bt = "not"
            pass

        if next_bt == "not" :
            add_data = collect_reviews(driver, nms)
            df = pd.concat([df, add_data], ignore_index = True)

            print(f"{nms}호텔 처음이자 마지막 페이지입니다. 총 페이지 수는 {review_page_cnt}입니다.")
            # driver.close() # 새탭 닫고 
            # driver.switch_to.window(driver.window_handles[0]) # 이전 창으로
            time.sleep(3)
            break

        if "inactive" in next_bt.get_attribute("class") :
            
            # 마지막페이지 데이터 가져오기
            add_data = collect_reviews(driver, nms) # 여기 예외처리해줘야돼. ***********
            df = pd.concat([df, add_data], ignore_index = True)

            print(f"{nms}호텔 마지막 페이지입니다. 총 페이지 수는 {review_page_cnt}입니다.")
            # driver.close() # 새탭 닫고 
            # driver.switch_to.window(driver.window_handles[0]) # 이전 창으로
            time.sleep(3)
            break

        else :

            try:
            # 먼저 다음 버튼을 클릭 시도
                add_data = collect_reviews(driver, nms)
                next_bt.click()
                df = pd.concat([df, add_data], ignore_index = True)

            except ElementClickInterceptedException:

            # 스크롤 다운
                driver.execute_script("window.scrollBy(0, 200);")
            # 다시 시도
                add_data = collect_reviews(driver, nms)
                next_bt.click()
                df = pd.concat([df, add_data], ignore_index = True)

            print(f"아직 {nms} 호텔 크롤링 진행중이에요. 현재 {review_page_cnt}페이지입니다.")
            review_page_cnt += 1
            
            time.sleep(3)

    # df.to_csv(f"C:/Users/UOS/proj_0/selenium/active-batch-3/active-batch-process-3.csv", encoding = 'utf-8-sig')

# sys.stdout.close()
