# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 09:28:19 2022

@author: junho
"""
import requests
import re
from bs4 import BeautifulSoup


# 정규표현식
# 패턴 형성
p = re.compile('ca.e')
# 패턴을 찾을 문자열 설정
m = p.match('case')
# 매칭된 문자열을 리턴  
m.group()

# .은 줄바꿈 문자 제외 모든 문자와 매칭 
p = re.compile('ca.e')
# 문자열과 매칭이 안됨
m = p.match('coffee')
# 문자열과 매칭이 안되서 에러가 나옴 
m.group()


def print_match(m):
    if m:
        print(m.group())
    else:
        print('Error')    


p = re.compile('ca.e')

m = p.match('care')
print_match(m)

m = p.match('caffee')
print_match(m)

# 시작이 ca로 시작하지 않아서 error 나옴 
m = p.match('good care')
print_match(m)


m = p.match('careless')
print_match(m)

type(p)


# 13:00~
# match랑 다름 serach는 찾는것 
# match는 처음부터 해당 문자가 있는가
# search는 그냥 포함하고 있는가

m1 = p.match('good care')
print_match(m1) 
m2 = p.search('good care')
print_match(m2)    



m1 = p.match('careless')
print_match(m1)    
m2 = p.search('careless')
print_match(m2)    


def print_match(m):
    if m:
        # 매칭된 문자열 
        print(f'm.group(): {m.group()}')
        # 입력받은 문자열
        print(f'm.string: {m.string}')
        # 매칭된 문자열의 시작 인덱스
        print(f'm.start(): {m.start()}')
        # 매칭된 문자열의 끝 인덱스
        print(f'm.end(): {m.end()}')
        # 시작, 끝 인덱스 
        print(f'm.span(): {m.span()}')
    else:
        print('Error')              

p = re.compile('ca.e')
m1 = p.search('good care')
print_match(m1)
print('='*80)
m2 = p.search('careless')
print_match(m2)


# findall 찾아서 리스트로 반환 
p.findall('careless')
p.findall('good care')
p.findall('good care care')





################ 웹스크래핑 ################
import requests
import re
from bs4 import BeautifulSoup
url = 'https://www.coupang.com/np/search?component=&q=%EB%85%B8%ED%8A%B8%EB%B6%81&channel=user'
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.81 Safari/537.36'
headers = {'User-Agent': user_agent}
res = requests.get(url, headers=headers)
res.raise_for_status

soup = BeautifulSoup(res.text, 'lxml')
soup.find_all('div', attrs={'class':'name'})

items = soup.find_all('li', attrs={'class': re.compile('^search-product')})
# print(items[0].find('div', attrs={'class':'name'}).get_text())
# items[0].find('div', attrs={'class':'name'}).text


for i in items:
    name = i.find('div', attrs={'class':'name'}).text
    name = name.split(', ')[0]
    price = i.find('strong', attrs={'class':'price-value'}).text
    rate = i.find('em', attrs={'class':'rating'})
    if rate:
        rate = rate.get_text()
    else:
        rate = '평점 없음'
    rate_count = i.find('span', attrs={'class':'rating-total-count'})
    if rate_count:
        rate_count = rate_count.get_text()
    else:
        rate = '평점 없음'
    print(name)
    print(price)
    print(rate)
    print(rate_count)
    print('='*80)



# 쿠팡 ad표시 상품 뽑기 
ad_list = []
for i in items:
    if i.find_all('span', attrs={'class':'ad-badge-text'}):
        name = i.find('div', attrs={'class':'name'}).text
        name = name.split(', ')[0]
        price = i.find('strong', attrs={'class':'price-value'}).text
        rate = i.find('em', attrs={'class':'rating'})
        if rate:
            rate = rate.get_text()
        else:
            rate = '평점 없음'
        rate_count = i.find('span', attrs={'class':'rating-total-count'})
        if rate_count:
            rate_count = rate_count.get_text()
        else:
            rate = '평점 없음'
        # print(name)
        # print(price)
        # print(rate)
        # print(rate_count)
        # print('='*80)
        ad_list.append([name, price, rate, rate_count])
    else:
        pass
ad_list    


# 쿠팡추천 표시 상품 뽑기 
coupang_list = []
for i in items:
    if i.find_all('img', attrs={'alt':'쿠팡추천'}):
        name = i.find('div', attrs={'class':'name'}).text
        name = name.split(', ')[0]
        price = i.find('strong', attrs={'class':'price-value'}).text
        rate = i.find('em', attrs={'class':'rating'})
        if rate:
            rate = rate.get_text()
        else:
            rate = '평점 없음'
        rate_count = i.find('span', attrs={'class':'rating-total-count'})
        if rate_count:
            rate_count = rate_count.get_text()
        else:
            rate = '평점 없음'
        # print(name)
        # print(price)
        # print(rate)
        # print(rate_count)
        # print('='*80)
        coupang_list.append([name, price, rate, rate_count])
    else:
        pass       
coupang_list 

# 여러 페이지에서 불러오기 
for i in range(1,6):
    url= f'https://www.coupang.com/np/search?q=%EB%85%B8%ED%8A%B8%EB%B6%81&channel=user&component=&eventCategory=SRP&trcid=&traid=&sorter=scoreDesc&minPrice=&maxPrice=&priceRange=&filterType=&listSize=36&filter=&isPriceRange=false&brand=&offerCondition=&rating=0&page={i}&rocketAll=false&searchIndexingToken=&backgroundColor='
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'lxml')
    items = soup.find_all('li', attrs={'class': re.compile('^search-product')})