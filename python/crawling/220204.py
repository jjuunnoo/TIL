# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 09:15:59 2022

@author: junho
"""
import requests
import os
from bs4 import BeautifulSoup


# response 객체로 받음
res = requests.get('http://www.naver.com')
# 200 정상처리 ok
# 404 not found / 403 forbidden
res.status_code

if res.status_code == requests.codes.ok:
    print('정상')
else:
    print('에러:'+ res.status_code)
    
    
# exception 강제 발생 raise
# 에러나면 알아서 exception 처리해줌 
res.raise_for_status()

res = requests.get('http://www.google.com')
res.raise_for_status()
print(res.text)
print(len(res.text))
os.getcwd()
lib = 'C:/Users/junho/Desktop/study/py/lib/'
os.chdir(lib)
os.getcwd()
with open ('result.html', 'w', encoding='utf-8') as f:
    f.write(res.text)    
    
    
    
# user agent 요즘엔 스크래핑 안막는데 막는 일부 사이트에 user agent 제공한다  
# 개발자도구의 console 창에 navigator.useragent 쓰면 확인 가능
# requests.get 할때 headers에 넣어준다 
url = 'http://blog.naver.com'
headers={
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.81 Safari/537.36'
    }   
res = requests.get(url, headers=headers)
res.raise_for_status()

with open ('agent.html','w',encoding = 'utf-8') as f:    
    f.write(res.text)
    
    

################## beatutifulsoup ##############
import requests
from bs4 import BeautifulSoup    
url = 'https://comic.naver.com/webtoon/weekday'
res=requests.get(url)
res.status_code
res.raise_for_status()    
# agent 정보를 다운로드, parser 다운받은것을 분석 
# requests로 부터 다운받은 정보를 res로 저장했고 안에 html 문서가 존재 
# beautifulsoup 사용을 위해 parser 진행 
soup = BeautifulSoup(res.text, 'lxml')
soup = BeautifulSoup(res.text, 'html.parser')



soup.title # > 꺽새는 &gt로 표현됨 
# 타이틀 태그 없이 텍스트만 가져오기 
soup.title.text
soup.title.get_text()

# 앵커 태그 찾기 값이 한개 밖에 안나옴  
soup.a

# soup.a는 dic
soup.a.attrs['href']
soup.a['href']

soup.find('a', attrs={'class':'Nbtn_upload'})
soup.find(attrs={'class':'Nbtn_upload'})



soup.find('li', attrs={'class':'rank01'})
rank1 = soup.find('li', attrs={'class':'rank01'})
rank1
# rank1은 li태그, 밑에 a태그의 title 
rank1.a.text
rank1.a.get_text()
rank1.a.attrs['title']


# 가족관계로 rank2를 찾아본다
# 예시의 경우 바로 아래가 아니라 하나 껴있어서 next_sibling 두개씀
rank2 = rank1.next_sibling.next_sibling
rank2.a
rank2.a.text
rank2.a.get_text()



rank3 = rank2.next_sibling.next_sibling
rank3.a.text

rank2 = rank3.previous_sibling.previous_sibling
rank2.a.text


# 부모태그 찾기
rank2.parent


# 동생찾기 원하는 태그를 넣어준다 
rank2 = rank1.find_next_sibling('li')
rank2.a.get_text()

rank3 = rank2.find_next_sibling('li')
rank3.a.text


# 동일수준 위태그 찾
rank2 = rank3.find_previous_sibling('li')
rank2.a.text

# 동일 수준 아래 태그 모두 찾기 
rank1.find_next_siblings('li')


url = 'https://comic.naver.com/webtoon/weekday'
res=requests.get(url)
res.raise_for_status()   
soup = BeautifulSoup(res.text, 'lxml')

# 전체 웹툰 목록 가져오기
cartoons = soup.find_all(
            'a', 
            attrs={'class':'title'}
            )
cartoons

for cartoon in cartoons:
    print(cartoon.get_text())


# 특정 웹툰에 대해 진행 
url = 'https://comic.naver.com/webtoon/list?titleId=748105&weekday=sun'
res = requests.get(url)
res.raise_for_status()
soup = BeautifulSoup(res.text,'lxml')


# 웹툰 화별로 제목 확인 title 
cartoons = soup.find_all('td', attrs={'class':'title'})
cartoons[0].a.text
cartoons[0].a.get_text()
cartoons[0].a['href']
cartoons[0].a['onclick']


# 링크잘려있어서 https://comic.naver.com 붙여주고 출력 
for cartoon in cartoons:
    title = cartoon.a.get_text()
    link = 'https://comic.naver.com' + cartoon.a['href']
    print(title, link)


# 웹툰 평균 평점 구해보기 
ratings = soup.find_all('div', attrs={'class':'rating_type'})
rate_sum = 0
for rating in ratings:
    rate = rating.find('strong').text
    rate_sum += float(rate)

round(rate_sum/len(ratings), 3)












