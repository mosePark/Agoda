import numpy as np
import pandas as pd

import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains # 스크롤 액션

chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
chrome_options.add_argument("--incognito")

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 광고 닫기 버튼                                                                 
def close_ad_popup():
    try:
        ad_buttons = driver.find_elements(By.CLASS_NAME, "ab-message-button")
        for bt in ad_buttons:
            if bt.text == '닫기':
                bt.click()
                break
    except NoSuchElementException:
        pass  # 광고가 없으면 무시                                        
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''    



driver = webdriver.Chrome(options=chrome_options)
url = 'https://www.agoda.com/?ds=yoqVTa5DmLnp6mR0'

driver.get(url)
#
close_ad_popup()
#

# 언어 변경
languge_button = driver.find_element(By.XPATH, "//*[@id='page-header']/section/div[2]/div[1]/div[2]/div[2]/div")
languge_button.click()

time.sleep(3)

eng_button = driver.find_element(By.CLASS_NAME, "avuwS")
eng_button.click()

time.sleep(3)
#
close_ad_popup()
#
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# ## 버튼 추천이 매일 달라짐 다른 카테고리에서 클릭하도록 !
# 통화 변경 배너 버튼
price_button = driver.find_element(By.XPATH, "//*[@id='page-header']/section/div[2]/div[1]/div[2]/div[1]/div")
price_button.click()

time.sleep(3)

# 달러 선택
dol_button = driver.find_element(By.XPATH, "//button[contains(., '미국 달러')]")
dol_button.click()

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 통화 변경
price_button = driver.find_element(By.XPATH, "//*[@id='page-header']/section/div[2]/div[1]/div[2]/div[1]/div")
price_button.click()

action = ActionChains(driver)

dol_button = driver.find_element(By.XPATH, "//button[contains(., 'US Dollar')]")
action.move_to_element(dol_button).click().perform()

#
close_ad_popup()
#
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 검색 엔진 박스
search_box = driver.find_element(By.ID, "textInput")

# 검색어 입력
loc_query = "manhattan"
search_box.send_keys(loc_query)
search_box.send_keys(Keys.ESCAPE)
# search_box.send_keys(Keys.ARROW_DOWN)
# search_box.send_keys(Keys.ENTER) # enter key 입력

# 검색버튼
search_button = driver.find_element(By.CSS_SELECTOR, "[data-selenium='searchButton']")
search_button.click()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
time_sleep(5)

crt_url = driver.current_url # 현재 url은?

last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # 페이지 끝까지 스크롤 내리기
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # 페이지 로드를 기다림 (2초 정도)
    time.sleep(2)

    # 새로운 높이를 가져와서 이전 높이와 비교
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# 호텔이름
hotel_elements = driver.find_elements(By.CSS_SELECTOR, "h3[data-selenium='hotel-name']")

# 리스트 넣기
hotel_nms = [hotel.text for hotel in hotel_elements]