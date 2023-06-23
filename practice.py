print("hello world")
hobby = "공놀"

#python 연산자 '/': 나누기(소수 표현), '//': 몫, '%': 나머지    

abs(-5) #절댓값
pow(4, 2) #4 ^ 2
max(4, 12) #큰값
min(4, 12) #작은값
round(3.14) #반올림

from math import * #*쓰면 math의 모든 기능을 사용하겠다는 의미

floor(4.99) #내림
ceil(4.9) #올리
sqrt(16) #제곱근

from random import *

print(random()) # 0.0 ~ 1.0 미만의 임의의 값 생성
print(random() * 10) # 0.0 ~ 10.0 미만의 임의의 값 생성
print(int(random())) # 0 ~ 1 미만의 임의의 값 생성  
print(int(random()) * 10) # 0 ~ 10 미만의 임의의 값 생성
print(int(random() * 10) + 1) # 1 ~ 10 미만의 임의의 값 생성

print(randrange(1, 46)) # 1 ~ 46 미만의 임의의 값 생성

print(randint(1, 45)) # 1~ ~ 45 이하의 임의의 값 생성

#문자열
son = "001109-1234567"

print("성별: " + son[7])
print("연: " + son[0:2]) # 0 부터 2 직전까지
print("월: " + son[2:4])
print("일: " + son[4:6])

print("생년월일: " + son[0:6]) # 처음부터 6직전까지
print("뒤 7자리: "+ son[7:]) # 7부터 끝까지
print("뒤 7자리 (뒤에서 부터): " + son[-7:]) # 맨 뒤에서 7번째부터 끝까지

#문자열 처리 함수
python = "python is Amazing"
print(python.lower())
print(python.upper())
print(python[0].isupper())
print(len(python))
print(python.replace("Python", "Java"))

index = python.index("n")
print(index)
index = python.index("n", index + 1)
print(index)

print(python.find("Java")) # if not true return -1
#print(python.index("Java")) # if not true return error

print(python.count("n")) # which place 'n' is at

#문자열 포맷

#방법 1
print("나는 %d살 입니다" % 20)
print("나는 %s을 좋아해요." % "파이썬")
print("Apple은 %c로 시작해요" % "A")

print("나는 %s색과 %s색을 좋아해요." % ("파란", "빨간"))

print("나는 {}살 입니다.".format(20))
print("나는 {age}살이며, {color}색을 좋아해요.".format(color = "빨간", age = 20))

age = 20
color = "빨간"

print(f"나는 {age}살이며 {color}색을 좋아해요")

# \n :줄바꿈
# \" \' : 문장 내에서 따옴표
# \\ : 문장 내에서 \
# \r : 커서를 맨 앞으로 이동
print("Red Apple\rPine") #Red Apple -> PineApple

 #\b : 백스페이스(한 글자 삭제)
print("Redd\bApple") #RedApple

# \t :탭
''' 사이트별로 비밀번호를 만들어 주는 프로그램을 작성하시오
    예) http://naver.com
    규칙1: http:// 부분은 제외 => naver.com
    규칙2: 처음 만나는 점(.) 이후 부분은 제외 => naver
    규칙3: 남은 글자중 처음 세자리 + 글자 개수 + 글자 내 'e' 개수 + "!"로 구성
    
    예) 생성된 비밀번호 : nav51!
'''


site = "http://naver.com"
my_str = site.replace("http://", "")
index = my_str.index('.')
my_str = my_str[:index]

answer = my_str[:3] + str(len(my_str)) + str(my_str.count("e")) + "!"
print(answer)

#List[]

subway = ["유", "조", "박"]
print(subway.index("조"))

subway.append("하")

print(subway)
subway.insert(1, "정")
print(subway)

print(subway.pop())
print(subway)

#정렬
num_list = [5,2,4,3,1]
num_list.sort()
print(num_list)

#순서 뒤집기
num_list.reverse()
print(num_list)

#모두 지우기
num_list.clear()

#다양한 자료형 함께 사용
mix_list = ["조세호", 1]

#리스트 확장
num_list.extend(mix_list)

#딕셔너리
cabinet = {3:"유재석", 100:"김태호"}
print(cabinet[3])
print(cabinet[100])
print(cabinet.get(3))

#print(cabinet[5]) #오류일어나고 프로그램 종료됨

print(cabinet.get(5)) #None 출력 후 이어짐
print(cabinet.get(5, "사용가능"))

print(3 in cabinet) #True
print(5 in cabinet) #False

#새손님
cabinet["A-3"] = "김종국"
cabinet["C-20"] = "조세호"

#간손님
del cabinet["A-3"]
print(cabinet)

#key 들만 출력
print(cabinet.keys())

#value 들만 출력
print(cabinet.values())

#key, value 쌍으로 출력
print(cabinet.items())

#목욕탕 폐점
cabinet.clear()


#튜플
#내용 변경 불가 but list보다 빠름

menu = ("돈가스", "치즈가스")
print(menu[0])
print(menu[1])

import numpy as np

a = np.ones((2,2), int)
#[[1 1]
# [1 1]]
b = np.full((2,2), 2)
#[[2 2]
# [2 2]]
c= np.array([1, 1])
#[1 1]
d= np.array([2, 2])
#[2 2]
print(np.concatenate((a,b), axis = 0))
print('-----------------------------')
print(np.concatenate((a,b), axis = 1))
print('-----------------------------')
print(np.stack((a,b), axis = 0))
print('-----------------------------')
print(np.stack((a,b), axis = 1))
print('-----------------------------')
print(np.stack((c,d), axis = 0))