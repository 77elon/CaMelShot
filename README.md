# CaMelShot
> 2022-1 캡스톤, OpenCV
  
  촬영하고자 하는 사진의 각도를 계산하고, 사전에 촬영된 사진를 기반으로 구도를 보정해주는 안드로이드 애플리케이션입니다.
  
  ![슬라이드9](https://user-images.githubusercontent.com/88618717/192153624-8c46ab3f-45a7-4af8-a7c0-99b3b590bf3b.PNG)

  * * * 

  ## 구현 환경
  OpenCV(CMake) -- 객체, 외곽선 검출 및 각도 계산
  Yolov5 (NDK) -- NCNN 기반 딥러닝 모델을 제작하였으며, 모바일 환경에서도 빠르게 건물을 감지 할 수 있습니다.

  ## 구현 기능

  * [사진 속 건물의 외곽선 검출]
  
  ![슬라이드10](https://user-images.githubusercontent.com/88618717/192153669-2b8b299a-adae-49d4-a6c7-75863188464c.PNG)
  ![슬라이드11](https://user-images.githubusercontent.com/88618717/192153689-c88c0027-68db-4e9f-9c3f-b0b4bca24abf.PNG)
  
  NCNN 모델로부터 객체의 위치를 추출하고, 객체의 외곽선 부분만을 OpenCV에서 처리합니다.
  
  * [개선된 외곽선 검출 기능]

  ![설명 1](https://user-images.githubusercontent.com/88618717/192153766-b3882d48-e3c9-483e-b46f-17ae7cde15c0.png)

  우리 CaMelShot은 사용자 친화적인 외곽선 검출을 위해, NCNN에서 추출된 건물의 넓이와 사진에서의 비율을 계산하여 가장 적합한 외곽선을 보여줍니다. 

  > 외곽선 검출 과정

  ![슬라이드 27](https://user-images.githubusercontent.com/88618717/192153779-b08fc582-d3d6-4d2e-b0e5-da5941c56d7f.png)
