# gaze vector g와 광고판 방향 벡터 d로
#   cos_sim = dot(g, d) / (||g|| ||d||)
#   angle = arccos(cos_sim)
#   angle <= threshold_deg 이면 “본 것”
# 여기까지는 OpenVINO 없이도 “더미 gaze”로 테스트 가능