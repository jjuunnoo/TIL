# selenium 동적페이지를 다룰때 
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

browser = webdriver.Chrome(r'C:\Users\junho\Desktop\study\py\chromedriver.exe')
browser.get('http://naver.com')


# browser.find_element_by_class_name('link_login').click()
# browser.find_elements(by='class', value='link_login').click()   # elements 는 리스트로 받아온다
browser.find_element(By.CLASS_NAME, 'link_login').click()
browser.back()
browser.forward()
browser.refresh()

from selenium.webdriver.common.keys import Keys

# query = browser.find_element_by_id('query')
query = browser.find_element(By.ID, 'query')
query.send_keys('네이버')
query.send_keys(Keys.ENTER)
tag = browser.find_element(By.TAG_NAME, 'a')
print(tag)

tag[0]
tags = browser.find_elements(By.TAG_NAME, 'a')

for tag in tags:
    tag.get_attribute('href')


browser = webdriver.Chrome(r'C:\Users\junho\Desktop\study\py\chromedriver.exe')
browser.get('http://daum.net')
# query = browser.find_element_by_name('q')
query = browser.find_element(By.NAME, 'q')
query.send_keys('다음')
query.send_keys(Keys.ENTER)

browser.find_element_by_xpath('//*[@id="daumSearch"]/fieldset/div/div/button[2]').click()
browser.find_element(By.XPATH, '//*[@id="daumSearch"]/fieldset/div/div/button[2]').click()
browser.quit()



