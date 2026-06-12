"""Editable test scenario for report_scenario_runner.py.

Structure is intentionally aligned with scenarios/default_accident.py:
- VIDEO_FILE
- SCENARIO = {"overview": ..., "details": ..., "workers": ...}
"""

VIDEO_FILE = "video/accident_video.mp4"

SCENARIO = {
    "overview": (
        "[사고 개요]\n"
        "- 현장: 테스트용 현장명을 입력하세요\n"
        "- 사업장: 테스트용 사업장 / 상시 근로자 수 입력\n"
        "- 사고 일시: 테스트용 사고 일시 입력\n"
        "- 사고 위치: 테스트용 사고 위치 입력\n"
        "- 사고 결과: 테스트용 사고 결과 입력"
    ),
    "details": (
        "[사고 경위]\n"
        "테스트용 사고 경위를 입력하세요.\n"
        "예: 비계 작업 중 작업발판 불안정으로 근로자가 추락함.\n"
        "예: 특별안전교육 미실시, 보호구 미착용, 안전난간 미설치 등 확인된 사실을 적으세요."
    ),
    "workers": (
        "[사업장 및 근로자 현황]\n"
        "- 재해자: 테스트용 재해자 정보 입력\n"
        "- 재해 결과: 부상 또는 사망 입력\n"
        "- 특별안전교육: 실시/미실시 입력\n"
        "- 작업 전 위험성평가: 실시/미실시 입력\n"
        "- 보호구: 착용/미착용 입력\n"
        "- 도급 관계: 해당 여부 입력"
    ),
}


