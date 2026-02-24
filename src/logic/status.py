# 프레임에서 바로 일/주/월을 만들기보다, 먼저 이벤트를 적재하고 집계하는 구조가 좋아.

# 추천 이벤트:
#   pass (유동 인구: ROI를 스쳐 지나감)
#   enter_roi, exit_roi
#   stay_start, stay_end (체류 시간 계산)
#   look_start, look_end (총 관심 시간 계산)

# 그리고 집계:
#   유동 인구 = 고유 track 수(혹은 pass 이벤트 수)
#   체류 인구 = stay_start 발생한 track 수
#   체류율 = 체류/유동
#   체류시간 평균/중앙값 = stay_end - stay_start 분포
#   관심 인구 = look_start 발생 track 수
#   관심률 = 관심/유동
#   총 관심 시간 = look 구간 합

# “프레임별로 누적”보다 “이벤트 기반”이 나중에 일/주/월로 묶기 훨씬 쉬워.