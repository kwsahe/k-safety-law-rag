"""Generate clean PNG diagrams for the RAG Report project."""

from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs"
FONT_REG = Path(r"C:\Windows\Fonts\malgun.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\malgunbd.ttf")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REG), size)


def rounded(draw: ImageDraw.ImageDraw, box, fill, outline, width=3, radius=20):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def centered(draw: ImageDraw.ImageDraw, box, text: str, fnt, fill="#162032", gap=7):
    x1, y1, x2, y2 = box
    lines = text.split("\n")
    sizes = [draw.textbbox((0, 0), line, font=fnt) for line in lines]
    widths = [right - left for left, top, right, bottom in sizes]
    heights = [bottom - top for left, top, right, bottom in sizes]
    total_h = sum(heights) + gap * (len(lines) - 1)
    y = y1 + ((y2 - y1) - total_h) / 2 - 2
    for line, width, height in zip(lines, widths, heights):
        draw.text((x1 + ((x2 - x1) - width) / 2, y), line, font=fnt, fill=fill)
        y += height + gap


def card(draw, x, y, w, h, text, *, fill="#ffffff", outline="#8da0ba", fnt=None):
    fnt = fnt or font(22)
    rounded(draw, (x, y, x + w, y + h), fill, outline, width=3, radius=18)
    centered(draw, (x + 18, y + 12, x + w - 18, y + h - 12), text, fnt)
    return (x, y, x + w, y + h)


def arrow(draw: ImageDraw.ImageDraw, start, end, *, color="#3d4c66", width=5):
    draw.line([start, end], fill=color, width=width)
    sx, sy = start
    ex, ey = end
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex >= sx else -1
        pts = [(ex, ey), (ex - 18 * direction, ey - 10), (ex - 18 * direction, ey + 10)]
    else:
        direction = 1 if ey >= sy else -1
        pts = [(ex, ey), (ex - 10, ey - 18 * direction), (ex + 10, ey - 18 * direction)]
    draw.polygon(pts, fill=color)


def label(draw, xy, text, size=19, fill="#526174", max_chars=36):
    x, y = xy
    for line in wrap(text, max_chars):
        draw.text((x, y), line, font=font(size), fill=fill)
        y += size + 7


def title(draw, text: str, subtitle: str):
    draw.text((80, 54), text, font=font(42, True), fill="#101827")
    draw.text((82, 112), subtitle, font=font(22), fill="#5b6475")


def make_flow_diagram():
    img = Image.new("RGB", (2200, 1450), "#f7f8fb")
    draw = ImageDraw.Draw(img)
    title(draw, "RAG Report 프로젝트 흐름도", "색인 생성과 질문 응답을 분리한 좌→우 처리 흐름")

    f_head = font(26, True)
    f_body = font(22)
    f_small = font(18)

    lanes = [
        (80, 190, 2120, 610, "#eef6ff", "#8fb8df", "1. 색인 생성 흐름"),
        (80, 700, 2120, 1300, "#f0f8f2", "#9bcba6", "2. 질문 응답 흐름"),
    ]
    for x1, y1, x2, y2, fill, outline, text in lanes:
        rounded(draw, (x1, y1, x2, y2), fill, outline, width=4, radius=28)
        draw.text((x1 + 30, y1 + 24), text, font=f_head, fill="#172033")

    # Ingestion lane: two parallel straight lines.
    card(draw, 150, 315, 260, 120, "data/laws\n법령 PDF + metadata", outline="#5c8fc4", fnt=f_body)
    card(draw, 560, 250, 350, 120, "scripts/run_ingest.py\nrag/ingest.py", outline="#5c8fc4", fnt=f_body)
    card(draw, 1080, 250, 300, 120, "chroma_db\n텍스트 법령 컬렉션", fill="#eaf3ff", outline="#5c8fc4", fnt=f_body)
    card(draw, 560, 440, 350, 120, "scripts/reingest_tables.py\nrag/table_retriever.py", outline="#5c8fc4", fnt=f_body)
    card(draw, 1080, 440, 300, 120, "chroma_db_tables\n표 법령 컬렉션", fill="#eaf3ff", outline="#5c8fc4", fnt=f_body)
    card(draw, 1580, 340, 360, 120, "저장된 검색 근거\n텍스트 + 표", fill="#ffffff", outline="#7b8aa4", fnt=f_body)

    arrow(draw, (410, 355), (560, 310))
    arrow(draw, (410, 355), (560, 500))
    arrow(draw, (910, 310), (1080, 310))
    arrow(draw, (910, 500), (1080, 500))
    arrow(draw, (1380, 310), (1580, 380), color="#7b8aa4", width=4)
    arrow(draw, (1380, 500), (1580, 420), color="#7b8aa4", width=4)
    label(draw, (575, 382), "PDF 로드, 청크, 임베딩", max_chars=30)
    label(draw, (575, 575), "pdfplumber 표 추출, 행/항목 청크, 임베딩", max_chars=40)

    # Query lane: a single straight main path.
    y = 910
    query_boxes = [
        (150, y, 250, 130, "사용자\n질문 + 사고 시나리오"),
        (500, y, 280, 130, "rag/chatbot.py\n질문/시나리오 조립"),
        (900, y, 330, 130, "rag/integrated_retriever.py\n텍스트+표 통합 검색"),
        (1350, y, 300, 130, "근거 컨텍스트\nPRIMARY / SECONDARY"),
        (1760, y, 270, 130, "qwen2.5:14b\nLLM 답변 생성"),
    ]
    for x, yy, w, h, text in query_boxes:
        card(draw, x, yy, w, h, text, outline="#51945f", fnt=f_body)
    for (x1, yy1, w1, h1, _), (x2, yy2, _w2, _h2, _text2) in zip(query_boxes, query_boxes[1:]):
        arrow(draw, (x1 + w1, yy1 + h1 // 2), (x2, yy2 + h1 // 2))

    card(draw, 1760, 1120, 270, 120, "최종 답변\n위반 판단 / 근거 조항", outline="#51945f", fnt=f_body)
    arrow(draw, (1895, 1040), (1895, 1120))
    arrow(draw, (1760, 1180), (400, 1180))
    label(draw, (905, 1095), "검색 단계에서 chroma_db와 chroma_db_tables를 함께 조회", max_chars=52)

    draw.text(
        (1525, 615),
        "질문 응답 단계에서 두 컬렉션을 조회합니다.",
        font=f_small,
        fill="#5c6574",
    )

    img.save(OUT_DIR / "project_flow_diagram.png")


def make_architecture_diagram():
    img = Image.new("RGB", (2200, 1450), "#fbfaf7")
    draw = ImageDraw.Draw(img)
    title(draw, "RAG Report 프로젝트 구성도", "화살표를 줄이고 계층·책임 중심으로 정리한 구조")

    f_head = font(26, True)
    f_body = font(21)
    f_small = font(18)

    groups = [
        (80, 200, 480, 1210, "#eef3ff", "#7fa0cf", "실행 진입점"),
        (540, 200, 1220, 1210, "#eff8f3", "#82b48d", "rag/ 코어 모듈"),
        (1280, 200, 1650, 1210, "#fff4e8", "#d9a05e", "데이터 / 저장소"),
        (1710, 200, 2120, 1210, "#f3eefc", "#a08bd6", "모델 / 출력"),
    ]
    for x1, y1, x2, y2, fill, outline, text in groups:
        rounded(draw, (x1, y1, x2, y2), fill, outline, width=4, radius=28)
        draw.text((x1 + 28, y1 + 24), text, font=f_head, fill="#172033")

    # Entry points.
    for i, text in enumerate(
        [
            "scripts/run_ingest.py\n텍스트 법령 색인",
            "scripts/reingest_tables.py\n표 법령 색인",
            "cli.py\n대화형 질문",
            "rag/table_pipeline.py\n표 검색/리포트",
            "dev_tools / tests\n추출·검색 검증",
        ]
    ):
        card(draw, 120, 310 + i * 145, 320, 95, text, fnt=f_body)

    # Core modules, grouped by responsibility.
    core_cards = [
        (585, 300, "config.py\n.env/경로 설정"),
        (880, 300, "schemas.py\n요청/응답 모델"),
        (585, 450, "ingest.py\n텍스트 청크"),
        (880, 450, "table_extraction.py\n표 추출"),
        (585, 600, "vector_store.py\n텍스트 DB 래퍼"),
        (880, 600, "table_chunking.py\n표 청크"),
        (585, 750, "retriever.py\n텍스트 검색"),
        (880, 750, "table_retriever.py\n표 검색/보정"),
        (585, 900, "integrated_retriever.py\n통합 랭킹"),
        (880, 900, "table_vector_store.py\n표 DB 래퍼"),
        (585, 1050, "chatbot.py\n프롬프트/LLM"),
        (880, 1050, "table_report.py\nHTML/PDF 리포트"),
    ]
    for x, y, text in core_cards:
        card(draw, x, y, 245, 90, text, fnt=f_body)

    # Data/storage.
    data_cards = [
        (1320, 320, "data/laws\nPDF + _metadata.json"),
        (1320, 520, "chroma_db\n텍스트 법령 컬렉션"),
        (1320, 720, "chroma_db_tables\n표 법령 컬렉션"),
        (1320, 920, "templates\nHTML 템플릿"),
    ]
    for x, y, text in data_cards:
        card(draw, x, y, 300, 95, text, outline="#c58d4a", fnt=f_body)

    # Models/output.
    model_cards = [
        (1750, 360, "BAAI/bge-m3\n임베딩 모델"),
        (1750, 610, "qwen2.5:14b\nLLM 추론 모델"),
        (1750, 860, "output/table_reports\n리포트 산출물"),
    ]
    for x, y, text in model_cards:
        card(draw, x, y, 330, 95, text, outline="#8d77c4", fnt=f_body)

    # Clean relationship summary: no crossing arrows inside the module map.
    summary_y = 1280
    summaries = [
        ("색인 생성", "법령 PDF → 텍스트/표 청크 → BAAI/bge-m3 임베딩 → ChromaDB 저장"),
        ("질문 응답", "질문·사고 시나리오 → 통합 검색 → 근거 컨텍스트 → qwen2.5:14b 답변"),
        ("리포트", "표 검색 결과 → HTML 템플릿 → output/table_reports"),
    ]
    for i, (head, body) in enumerate(summaries):
        x = 120 + i * 680
        rounded(draw, (x, summary_y, x + 600, summary_y + 95), "#ffffff", "#d2d8e3", width=2, radius=16)
        draw.text((x + 24, summary_y + 20), head, font=font(20, True), fill="#172033")
        draw.text((x + 24, summary_y + 52), body, font=f_small, fill="#526174")

    draw.text(
        (560, 1230),
        "읽는 방법: 왼쪽은 실행 명령, 가운데는 구현 책임, 오른쪽은 데이터·모델·산출물입니다.",
        font=f_small,
        fill="#596273",
    )

    img.save(OUT_DIR / "project_architecture_diagram.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_flow_diagram()
    make_architecture_diagram()
    print(OUT_DIR / "project_flow_diagram.png")
    print(OUT_DIR / "project_architecture_diagram.png")


if __name__ == "__main__":
    main()
