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

# 스크롤 천천히 내리기
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

# 다시 맨위로
driver.execute_script("window.scrollTo(0, 0);")

# 호텔 클릭하고 각각 원하는 데이터 가져오기 *****************

cols = ['Score', 'Country', 'Traveler Type', 'Room Type', 'Stay Duration', 'Title', 'Text', 'Date']
df = pd.DataFrame(columns = cols)

cnt = 1
target_numbs = 100

for htnm in hotel_nms[0:3] : # 3개만 예시로 돌려보자.
    
    # 데이터 긁어오면 이전에 호텔 보여줬던 그 페이지로 다시 돌아와야해.
    
    hotel_list_url = driver.current_url

    while True :
        try : 
            hotel_bt = driver.find_element(By.XPATH, f"//h3[@data-selenium='hotel-name' and text()='{hotel_name}']")
            hotel_bt.click()
            url_htnm = driver.current_url
            break

        except NoSuchElementException :
            driver.execute_script("window.scrollBy(0, 500);")
            time.sleep(3)

    time.sleep(3)
    
    next_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".ficon-carrouselarrow-right"))) # 이거 리뷰넘기는거임

    try:
        while True:
            # 현재 페이지의 리뷰 데이터 수집
            review_elements = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "Review-comment"))
            )

            for review_element in review_elements:
                
                reviewer_elem = review_element.find_elements(By.CLASS_NAME, "Review-comment-reviewer")
                
                score = review_element.find_element(By.CLASS_NAME, "Review-comment-leftScore").text
                country = review_element.find_element(By.CLASS_NAME, "flag").get_attribute("class").split("-")[-1].upper()
                traveler_type = reviewer_elem[1].text if len(reviewer_elem) > 1 else None
                room_type = reviewer_elem[2].text if len(reviewer_elem) > 2 else None
                stay_duration = reviewer_elem[3].text if len(reviewer_elem) > 3 else None
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

        driver.get(hotel_list_url) # 호텔리스트 url로
            
    except Exception as e:
        print("Error:", e)

df.to_csv("hotels-data-100.csv", encoding = 'utf-8')