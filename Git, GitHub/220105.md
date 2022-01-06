# Git / GitHub

- 분산 버전 관리 시스템
- 문서 수정, 코드 작성 시 히스토리(버전) 관리
- 개발 과정, 변경사항 트래킹
- 백업, 복구, 협업



# GUI / CLI

- `GUI` : graphic user interface, 그래픽을 통해 상호작용
  - CLI에 비해 성능 많이 소모
- `CLI` : command line interface, 터미널을 통해 상호작용
  - GUI로 불가능한 세부적인 기능 사용



# Git Bash

- 명령어 통일
  - UNIX 운영체제와 Windows의 명령어 차이
  - Git Bash  일종의 번역기
  - Windows에서 UNIX 계열 운영체제 터미널 사용 위함
  - 개발자 입장, 개발시장에서 UNIX 계열 운영체제가 더 많이 사용됨



# 터미널 명령어

- `mkdir` : 폴더 생성

  - ```bash
    mkdir name
    mkdir 'name space' #띄어쓰기 포함
    mkdir name1 name2 #여러개 동시 생성
    ```

- `touch` : 파일 생성

  - ```bash
    touch name
    touch name1 name2 #여러개 동시 생성
    touch .name #숨김 파일
    ```

- `ls` : 현재 디렉토리의 폴더/파일 목록 확인

  - `-a` : all, 숨김 파일까지
  - `-l` : long, 용량, 수정 날짜 등의 자세한 정보

- `cd`: 작업 중인 디렉토리 변경

  - ```bash
    cd folder #작업중인 디렉토리에 있는 폴더로 이동
    cd ~ #홈
    cd .. #위로 가기
    cd - #뒤로 가기
    ```

- `rm` : 폴더/파일 삭제

  - ```bash
    rm -r foldername #폴더 삭제
    rm filename #파일 삭제
    ```

- `mv` : 파일 이동, 이름 변경

  - ```bash
    mv filename foldername #파일을 폴더로 이동
    mv filename changename #파일 이름 변경
    ```




# Git 기초 / 명령어

- 로컬 저장소

  - `Working directory` : 사용자의 일반적인 작업 공간
  - `Staging Area` : 커밋을 위한 파일 및 폴더 추가 공간
  - `Repository` : staging area 의 변경사항(커밋) 저장

- Git - Github 연결 (user) : 한번만 하면 됨

  - `git config`

    ```bash
    git config --global user.name <username>  #username 등록
    git config --global user.name  #username 확인
    git config --global user.email <email>  #email 등록
    git config --global user.email  #email 확인
    ```

- Git - Github 연결 (Repository) : Repository 연결 시 한번만

  - `git init` : git 시작

    ```bash
    git init
    ```

    - `.git` 숨김폴더 생성, 터미널에 `master` 표기

  - `git remote` : repository 연결

    ```bash
    git remote add <이름> <repository url>
    git remote -v #원격 저장소 조회
    git remote rm <이름> #원격 저장소 연결 삭제
    git remote remove <이름> #원격 저장소 연결 삭제
    ```

- `git status` : Working directory와 Staging Area에 있는 파일의 현재 상태 확인

  ```bash
  git status
  ```

  - Untracked : Git이 관리하지 않는 파일
  - Tracked : Git이 관리하는 파일
    - Unmodified : 최신상태
    - Modifiec : 수정되었으나 Staging Area에 미반영
    - Staged : Staging Area에 올라간 상태

- `git add` : Working directory에 있는 파일을 Staging Area로 올림

  ```bash
  git add a.txt #특정 파일
  git add folder/ #특정 폴더
  git add . #현재 디렉토리에 속한 파일/폴더 전부
  ```

  - Git이 트래킹 할 수 있도록 함

- `git commit` : 커밋 메시지 작성

  ```bash
  git commit -m 'commit'
  ```

  - Staging Area에 올라온 파일의 변경 사항을 하나의 버전(커밋)으로 저장
  - 어떤 수정사항이 있는지 메모

- `git log`

  ```bash
  git log
  ```

  - 커밋의 내역(ID, 작성자, 시간, 메시지)을 조회
    - `--oneline` : 한 줄로 축약
    - `--graph` : 그래프
    - `--all` : 모든 브랜치의 내역
    - `--reverse` : 순서 반대로(최신이 가장 아래)
    - `-p` : 파일 변경 내용 포함
    - `-숫자` : 원하는 갯수 만큼

- `git push`

  ```bash
  git push <저장소 이름> <브랜치 이름>
  ```

  - `-u` : 두 번째 커밋부터는 저장소, 브랜치 이름 생략 가능

    ```bash
    git push -u <저장소 이름> <브랜치 이름> 
    git push #-u 이후 축약사용
    ```

    

