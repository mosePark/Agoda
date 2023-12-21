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
chrome_options.add_argument("--incognito") # 시크릿 모드 활성화


driver = webdriver.Chrome(options=chrome_options)

url = 'https://www.agoda.com/ko-kr/hilton-times-square-hotel/hotel/new-york-ny-us.html?finalPriceView=1&isShowMobileAppPrice=false&cid=1891463&numberOfBedrooms=&familyMode=false&adults=2&children=0&rooms=1&maxRooms=0&checkIn=2023-12-14&isCalendarCallout=false&childAges=&numberOfGuest=0&missingChildAges=false&travellerType=1&showReviewSubmissionEntry=false&currencyCode=KRW&isFreeOccSearch=false&tag=45b17d1d-e0b0-fe2a-ce90-5513829d856b&isCityHaveAsq=false&los=1&searchrequestid=48d4bfa9-c94f-4230-b293-31bdcf6e97c7&ds=EQDe0O9a%2FbZzOyX5'

driver.get(url) # 브라우저 가져와!

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# # 원하는 요소가 로드될 때까지 기다림
# try:
#     # "Review-comment" 클래스를 가진 모든 요소를 찾음
#     review_elements = WebDriverWait(driver, 10).until(
#         EC.presence_of_all_elements_located((By.CLASS_NAME, "Review-comment"))
#     )

#     for review_element in review_elements:
#         # 왼쪽 섹션 (점수, 나라 등) 추출
#         score = review_element.find_element(By.CLASS_NAME, "Review-comment-leftScore").text
#         country = review_element.find_element(By.CLASS_NAME, "flag").get_attribute("class").split("-")[-1].upper()
#         traveler_type = review_element.find_elements(By.CLASS_NAME, "Review-comment-reviewer")[1].text
#         room_type = review_element.find_elements(By.CLASS_NAME, "Review-comment-reviewer")[2].text
#         stay_duration = review_element.find_elements(By.CLASS_NAME, "Review-comment-reviewer")[3].text

#         # body (리뷰 본문) 추출
#         title = review_element.find_element(By.CLASS_NAME, "Review-comment-bodyTitle").text
#         text = review_element.find_element(By.CLASS_NAME, "Review-comment-bodyText").text
#         date = review_element.find_element(By.CLASS_NAME, "Review-statusBar-date").text

#         print("Score:", score)
#         print("Country:", country)
#         print("Traveler Type:", traveler_type)
#         print("Room Type:", room_type)
#         print("Stay Duration:", stay_duration)
#         print("Title:", title)
#         print("Text:", text)
#         print("Date:", date)
#         print("------------------------------------")

# except Exception as e:
#     print("Error:", e)

# finally:
#     # 브라우저 닫기
#     driver.quit()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# # 몇 번째 페이지인지 찾기 (현재 페이지 번호 찾기)
# current_page_number = int(driver.find_element(By.CSS_SELECTOR, ".Review-paginator-number--current").text)
# print(current_page_number)

# # 다음 페이지 번호 계산
# next_page_number = current_page_number + 1

# # 다음 페이지 버튼 찾기
# next_page_button = WebDriverWait(driver, 10).until(
#     EC.element_to_be_clickable((By.XPATH, f"//span[contains(@class, 'Review-paginator-number') and text()='{next_page_number}']"))
# )

# # 다음 페이지 버튼 클릭
# next_page_button.click()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# elem = driver.find_element(By.role, "link")
# elem.click()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# # 페이지 다음으로 넘어가기 코드!
# # Wait time
# wait = WebDriverWait(driver, 10)

# next_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".ficon-carrouselarrow-right")))

# for i in range(1,5) : 
#     next_button.click()

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# import pandas as pd
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

# # 브라우저 꺼짐 방지
# chrome_options = Options()
# chrome_options.add_experimental_option("detach", True)

# # 빈 데이터프레임 생성
# columns = ['Score', 'Country', 'Traveler Type', 'Room Type', 'Stay Duration', 'Title', 'Text', 'Date']
# df = pd.DataFrame(columns=columns)

# # 웹 드라이버 설정
# browser = webdriver.Chrome(options=chrome_options)

# # 웹사이트 열기
# url = 'https://www.agoda.com/ko-kr/hilton-times-square-hotel/hotel/new-york-ny-us.html?...'
# browser.get(url)

# try:
#     while True:
#         # 현재 페이지의 리뷰 데이터 수집
#         review_elements = WebDriverWait(browser, 10).until(
#             EC.presence_of_all_elements_located((By.CLASS_NAME, "Review-comment"))
#         )

#         for review_element in review_elements:
#             # 리뷰 세부 정보 추출 후 데이터프레임에 추가하는 로직은 기존과 동일

#         # 다음 페이지 버튼 찾기
#             next_button = WebDriverWait(browser, 10).until(
#             EC.presence_of_element_located((By.CSS_SELECTOR, ".ficon-carrouselarrow-right"))
#         )

#         # 다음 페이지 버튼이 비활성화되어 있으면 루프 종료
#         if "disabled" in next_button.get_attribute("class"):
#             break

#         # 다음 페이지로 이동
#         next_button.click()

# except Exception as e:
#     print("Error:", e)

# finally:
#     browser.quit()

# # 데이터프레임 출력
# print(df)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# current_page = 1
# while True:
#     # Process reviews on the current page...

#     # Attempt to go to the next page by number
#     next_page = current_page + 1
#     try:
#         next_page_button = WebDriverWait(driver, 10).until(
#             EC.presence_of_element_located((By.XPATH, f"//span[@role='link'][text()='{next_page}']"))
#         )
#         next_page_button.click()
#         current_page += 1
#     except Exception as e:
#         # If specific page number is not available, try clicking the next arrow
#         next_button = WebDriverWait(driver, 10).until(
#             EC.presence_of_element_located((By.CSS_SELECTOR, ".ficon-carrouselarrow-right"))
#         )
#         if "disabled" in next_button.get_attribute("class"):
#             break
#         next_button.click()
#         print(current_page)

# # Remember to close the driver after the loop
# # driver.quit()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# next_button = WebDriverWait(driver, 10).until(
#     EC.presence_of_element_located((By.CSS_SELECTOR, ".ficon-carrouselarrow-right"))
#     )
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# while True:
#     next_page_button = driver.find_element(By.CSS_SELECTOR, 'span[aria-label="다음 이용후기 페이지"]')
        
#     # 버튼이 비활성화 상태인지 확인
#     if 'inactive' in next_page_button.get_attribute('class'):
#         break  # 비활성화되어 있다면 반복 중단
    
#     # 다음 페이지로 이동
#     next_page_button.click()
#     time.sleep(2)  # 페이지 로딩 대기
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 이동하려는 페이지 번호 설정 ********************************************************************************
# target_page = 5

# while True:

#     # 현재 페이지 번호를 찾습니다.
#     current_page_element = driver.find_element(By.CLASS_NAME, 'Review-paginator-number--current')
#     current_page_number = int(current_page_element.text)

#     # 목표 페이지에 도달했는지 확인
#     if current_page_number == target_page:
#         print(f"현재 페이지 {current_page_number}에 도달했습니다.")
#         break

#     # 목표 페이지로 이동
#     target_page_element = driver.find_element(By.XPATH, f"//span[@class='Typographystyled__TypographyStyled-sc-j18mtu-0 cfuHGb kite-js-Typography Review-paginator-number ' and text()='{target_page}']")
#     target_page_element.click()
#     time.sleep(2)  # 페이지 로딩 대기
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# # 페이지네이션의 모든 페이지 번호를 찾습니다.
# page_numbers = driver.find_elements(By.CSS_SELECTOR, '.Review-paginator-numbers span[class*="kite-js-Typography Review-paginator-number"]')

# # 페이지 번호 목록에서 마지막 요소의 텍스트를 추출합니다.
# last_page_number = int(page_numbers[-1].text)
# print(f"마지막 페이지 번호: {last_page_number}")
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# html_content = driver.page_source
# print(html_content)

# next_page_button = driver.find_element(By.CSS_SELECTOR, "span.Review-paginator-arrow:not(.Review-paginator-arrow--inactive)")
# next_page_button.click()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
driver.get(url)

def get_current_and_total_pages(driver):
    try:
        page_bar = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="review_list_page_container"]/div[4]/div'))
        )
        current_page_element = page_bar.find_element(By.CLASS_NAME, 'bui-pagination__item--active')
        current_page_text = current_page_element.text
        if 'Current' in current_page_text:
            current_page = int(current_page_text.split()[-1])
        else:
            current_page = int(current_page_text)
        
        total_pages_element = page_bar.find_elements(By.CLASS_NAME, 'bui-pagination__link')[-1]
        total_pages_text = total_pages_element.text
        total_pages = int(total_pages_text.split()[-1])
        return current_page, total_pages
    except NoSuchElementException:
        return None, None

current_page, total_pages = get_current_and_total_pages(driver)
print(f"Current Page: {current_page}, Total Pages: {total_pages}")
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''