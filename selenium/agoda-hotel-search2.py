# 선형대수
import numpy as np
import pandas as pd

# 타임
import time

# 동적크롤링
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains # 스크롤 액션

# 예외클래스 임포트
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException

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
def slow_scroll_to_end(driver, scroll_step=200, delay=0.5):
    """ 페이지를 천천히 스크롤 내리는 함수.

    :param driver: Selenium WebDriver 객체.
    :param scroll_step: 한 번에 스크롤할 픽셀 수.
    :param delay: 스크롤 사이의 지연 시간(초).
    """
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    while True:
        # 현재 높이에서 scroll_step만큼 아래로 스크롤
        driver.execute_script(f"window.scrollBy(0, {scroll_step});")

        # 페이지 로드를 위한 지연 시간
        time.sleep(delay)

        # 새로운 높이 계산
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break  # 스크롤이 끝에 도달하면 종료
        last_height = new_height
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# def slow_scroll_until_next_visible(driver, scroll_step=200, delay=0.5, timeout=10):
#     """ 'Next' 버튼이 보일 때까지 페이지를 천천히 스크롤하는 함수.

#     :param driver: Selenium WebDriver 객체.
#     :param scroll_step: 한 번에 스크롤할 픽셀 수.
#     :param delay: 스크롤 사이의 지연 시간(초).
#     :param timeout: 'Next' 버튼의 가시성 확인을 위한 최대 대기 시간(초).
#     """
#     last_height = driver.execute_script("return document.body.scrollHeight")
#     next_visible = False

#     while not next_visible:
#         # 현재 높이에서 scroll_step만큼 아래로 스크롤
#         driver.execute_script(f"window.scrollBy(0, {scroll_step});")
#         time.sleep(delay)

#         # 새로운 높이 계산
#         new_height = driver.execute_script("return document.body.scrollHeight")
#         if new_height == last_height:
#             # 더 이상 스크롤할 내용이 없으면 중단
#             break
#         last_height = new_height

#         # 'Next' 버튼의 가시성 확인
#         try:
#             WebDriverWait(driver, timeout).until(
#                 EC.visibility_of_element_located((By.ID, "paginationNext"))
#             )
#             next_visible = True
#         except TimeoutException:
#             continue
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def slow_scroll_to_top(driver, scroll_step=-200, delay=0.5):

    current_height = driver.execute_script("return window.pageYOffset")

    while True:
        driver.execute_script(f"window.scrollBy(0, {scroll_step});")
        
        time.sleep(delay)

        new_height = driver.execute_script("return window.pageYOffset")
        if new_height >= current_height or new_height <= 0:
            break
        current_height = new_height
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
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
#
close_ad_popup()
#

eng_button = driver.find_element(By.CLASS_NAME, "avuwS")
eng_button.click()

time.sleep(3)
#
close_ad_popup()
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
#
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 검색 엔진 박스
search_box = driver.find_element(By.ID, "textInput")

# 검색어 입력
loc_query = "manhattan"
search_box.send_keys(loc_query)

time.sleep(3)

search_box.send_keys(Keys.ESCAPE)
# search_box.send_keys(Keys.ARROW_DOWN)
# search_box.send_keys(Keys.ENTER) # enter key 입력

time.sleep(3)
#
close_ad_popup()
#

# 검색버튼
search_button = driver.find_element(By.CSS_SELECTOR, "[data-selenium='searchButton']")
search_button.click()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
time.sleep(3)
#
close_ad_popup()
#


# 약 1분 30초
before_loc = driver.execute_script("return window.pageYOffset")
scroll_amount = driver.execute_script("return document.body.scrollHeight") / 10

while True:
    driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
    time.sleep(2)

    after_loc = driver.execute_script("return window.pageYOffset")

    if before_loc == after_loc:
        break
    else:
        before_loc = after_loc

time.sleep(3)
#
close_ad_popup()
#

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 호텔이름
hotel_elements = driver.find_elements(By.CSS_SELECTOR, "h3[data-selenium='hotel-name']")

# 리스트 넣기

hotel_nms = [hotel.text for hotel in hotel_elements]
len(hotel_nms)