# 실행방법: python -m tests.sue.test_status

from src.logic.status import StatusTracker
from src.utils.types import (
    FrameMeta,
    Track,
    BBoxXYXY,
    LookResult,
    ROI,
    PersonAttr,
    Gender,
    AgeGroup,
)

def make_meta(frame_idx: int, ts_ms: int, fps: float = 30.0):
    return FrameMeta(
        frame_idx=frame_idx,
        ts_ms=ts_ms,
        fps=fps,
        width=1280,
        height=720,
    )

def make_track(
    track_id: int,
    looking: bool,
    in_roi: bool = True,
    age_group=AgeGroup.young,
    gender=Gender.female,
):
    return Track(
        track_id=track_id,
        bbox=BBoxXYXY(100, 100, 200, 300),
        attr=PersonAttr(gender=gender, age_group=age_group),
        roi=ROI(in_roi=in_roi, dwell_frames=1),
        look_result=LookResult(
            is_looking=looking,
            score=0.9 if looking else 0.2,
            angle_deg=10.0 if looking else 50.0,
        ),
    )

def test_status_tracker():
    status = StatusTracker()

    # frame 0: 등장, 안 봄
    meta0 = make_meta(0, 0)
    tracks0 = [make_track(track_id=1, looking=False)]
    status.update(meta0, tracks0)

    # frame 1: 보기 시작
    meta1 = make_meta(1, 1000)
    tracks1 = [make_track(track_id=1, looking=True)]
    status.update(meta1, tracks1)

    # frame 2: 계속 봄
    meta2 = make_meta(2, 2000)
    tracks2 = [make_track(track_id=1, looking=True)]
    status.update(meta2, tracks2)

    # frame 3: 보기 종료
    meta3 = make_meta(3, 3000)
    tracks3 = [make_track(track_id=1, looking=False)]
    status.update(meta3, tracks3)

    # frame 4: track 사라짐
    meta4 = make_meta(4, 4000)
    tracks4 = []
    status.update(meta4, tracks4)

    status.finalize()
    results = status.get_results()

    print(results)

    assert len(results) == 1
    assert results[0]["track_id"] == 1
    assert results[0]["age_group"] == "young_adult"
    assert results[0]["gender"] == "female"
    assert results[0]["total_look_duration_ms"] == 2000
    assert len(results[0]["look_times"]) == 1

if __name__ == "__main__":
    test_status_tracker()