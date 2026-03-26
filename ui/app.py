# ui/app.py
import os
import json
import html
import streamlit as st
import requests

st.set_page_config(page_title="RAG — PDF & Image OCR (UI)", layout="wide")

BACKEND = os.environ.get("BACKEND_URL", "http://rag-backend:8000")

st.markdown(
    "\n".join(
        [
            "<style>",
            "    :root {",
            "        --surface: #ffffff;",
            "        --surface-muted: #f3f6fb;",
            "        --border: #e3e8f1;",
            "        --accent: #0059ff;",
            "        --accent-soft: rgba(90, 128, 255, 0.12);",
            "        --text-muted: #5f6c86;",
            "        --success: #1dad6d;",
            "        --danger: #ff4d4d;",
            "    }",
            "    .page-lead {",
            "        font-size: 1rem;",
            "        color: #4a5568;",
            "        margin: 0 0 1.5rem;",
            "    }",
            "    .status-card {",
            "        display: flex;",
            "        align-items: flex-start;",
            "        gap: 0.85rem;",
            "        border-radius: 1.25rem;",
            "        padding: 1.4rem;",
            "        background: var(--surface);",
            "        border: 1px solid var(--border);",
            "        box-shadow: 0 25px 55px rgba(15, 23, 42, 0.08);",
            "        height: 100%;",
            "    }",
            "    .status-card .status-icon {",
            "        width: 50px;",
            "        height: 50px;",
            "        border-radius: 1rem;",
            "        background: var(--surface-muted);",
            "        display: flex;",
            "        align-items: center;",
            "        justify-content: center;",
            "        font-size: 1.4rem;",
            "        color: #1f2937;",
            "    }",
            "    .status-card .status-meta { flex: 1; }",
            "    .status-card .label {",
            "        font-size: 0.85rem;",
            "        letter-spacing: 0.02rem;",
            "        color: var(--text-muted);",
            "        text-transform: uppercase;",
            "    }",
            "    .status-card .value {",
            "        font-size: 1.1rem;",
            "        font-weight: 600;",
            "        margin-top: 0.2rem;",
            "    }",
            "    .status-card .helper {",
            "        font-size: 0.85rem;",
            "        color: var(--text-muted);",
            "        margin-top: 0.35rem;",
            "    }",
            "    .status-card .status-pill {",
            "        display: inline-flex;",
            "        align-items: center;",
            "        gap: 0.35rem;",
            "        padding: 0.2rem 0.85rem;",
            "        border-radius: 999px;",
            "        background: var(--accent-soft);",
            "        color: var(--accent);",
            "        font-size: 0.85rem;",
            "        font-weight: 600;",
            "        margin-top: 0.5rem;",
            "    }",
            "    .status-card.highlight {",
            "        color: #fff;",
            "        border: none;",
            "        box-shadow: 0 30px 65px rgba(3, 7, 18, 0.25);",
            "    }",
            "    .status-card.highlight.online {",
            "        background: linear-gradient(135deg, #0d925f, #1fc182);",
            "    }",
            "    .status-card.highlight.offline {",
            "        background: linear-gradient(135deg, #b42318, #ff7870);",
            "    }",
            "    .status-card.highlight .status-icon {",
            "        background: rgba(255, 255, 255, 0.15);",
            "        color: #fff;",
            "    }",
            "    .status-card.highlight .label { color: rgba(255, 255, 255, 0.85); }",
            "    .status-card.highlight .status-pill {",
            "        background: rgba(255, 255, 255, 0.18);",
            "        color: #fff;",
            "    }",
            "    .status-dot {",
            "        width: 12px;",
            "        height: 12px;",
            "        border-radius: 999px;",
            "        background: var(--success);",
            "        display: inline-flex;",
            "    }",
            "    .status-card.highlight.offline .status-dot { background: var(--danger); }",
            "    .info-card {",
            "        border-radius: 1rem;",
            "        padding: 1.5rem;",
            "        background: var(--surface);",
            "        border: 1px solid var(--border);",
            "        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.4), 0 15px 35px rgba(15, 23, 42, 0.08);",
            "        margin-top: 1.2rem;",
            "        margin-bottom: 2.4rem;",
            "    }",
            "    .chips {",
            "        display: flex;",
            "        gap: 0.5rem;",
            "        flex-wrap: wrap;",
            "        margin-bottom: 0.7rem;",
            "    }",
            "    .chip {",
            "        padding: 0.3rem 0.8rem;",
            "        border-radius: 999px;",
            "        background: var(--surface-muted);",
            "        font-size: 0.85rem;",
            "        color: #1f2937;",
            "    }",
            '    .stTabs [data-baseweb="tab"] {',
            "        padding: 0.5rem 1.2rem;",
            "        border-radius: 999px;",
            "        background: var(--surface-muted);",
            "        color: #1f2a37;",
            "    }",
            '    .stTabs [aria-selected=\"true\"] {',
            "        background: #1f3d8f !important;",
            "        color: #fff !important;",
            "    }",
            "    .backend-label {",
            "        font-size: 0.85rem;",
            "        letter-spacing: 0.05rem;",
            "        text-transform: uppercase;",
            "        color: var(--text-muted);",
            "        margin-bottom: 0.25rem;",
            "    }",
            "    .backend-url {",
            "        font-weight: 600;",
            "        font-size: 1.05rem;",
            "        margin-bottom: 0.8rem;",
            "        word-break: break-all;",
            "        color: #0f172a;",
            "    }",
            "    .backend-helper {",
            "        font-size: 0.85rem;",
            "        color: var(--text-muted);",
            "        margin-bottom: 1rem;",
            "    }",
            "    .backend-slider-label {",
            "        font-size: 0.9rem;",
            "        font-weight: 600;",
            "        margin-bottom: 0.2rem;",
            "    }",
            "    .slider-caption {",
            "        font-size: 0.85rem;",
            "        color: var(--text-muted);",
            "        margin-top: 0.45rem;",
            "    }",
            "    div.top-flag, div.bottom-flag, div.backend-card-flag {",
            "        width: 0;",
            "        height: 0;",
            "        padding: 0;",
            "        margin: 0;",
            "    }",
            "    div[data-testid=\"stVerticalBlock\"]:has(> div.top-flag),",
            "    div[data-testid=\"stVerticalBlock\"]:has(> div.bottom-flag) {",
            "        min-height: 45vh;",
            "        display: flex;",
            "        flex-direction: column;",
            "        justify-content: center;",
            "    }",
            "    div[data-testid=\"stVerticalBlock\"]:has(> div.bottom-flag) {",
            "        justify-content: flex-start;",
            "    }",
            "    div[data-testid=\"stVerticalBlock\"]:has(> div.backend-card-flag) {",
            "        border-radius: 1.25rem;",
            "        padding: 1.5rem 1.6rem 1.2rem;",
            "        background: var(--surface);",
            "        border: 1px solid var(--border);",
            "        box-shadow: 0 25px 55px rgba(15, 23, 42, 0.08);",
            "    }",
            "    div[data-testid=\"stVerticalBlock\"]:has(> div.backend-card-flag) .stSlider {",
            "        padding-top: 0.2rem;",
            "    }",
            "    .result-card {",
            "        border-radius: 1.2rem;",
            "        padding: 1.4rem 1.6rem;",
            "        background: var(--surface);",
            "        border: 1px solid var(--border);",
            "        box-shadow: 0 25px 55px rgba(15, 23, 42, 0.08);",
            "    }",
            "    .result-card .result-pill {",
            "        display: inline-flex;",
            "        align-items: center;",
            "        gap: 0.4rem;",
            "        padding: 0.3rem 1rem;",
            "        border-radius: 999px;",
            "        font-weight: 600;",
            "        font-size: 0.9rem;",
            "    }",
            "    .result-card .result-pill.ok {",
            "        background: rgba(29, 173, 109, 0.18);",
            "        color: #10734a;",
            "    }",
            "    .result-card .result-pill.warn {",
            "        background: rgba(255, 160, 160, 0.25);",
            "        color: #9f1d1d;",
            "    }",
            "    .result-card .result-stats {",
            "        display: flex;",
            "        gap: 1.8rem;",
            "        flex-wrap: wrap;",
            "        margin-top: 1rem;",
            "    }",
            "    .result-card .result-stat .label {",
            "        font-size: 0.85rem;",
            "        color: var(--text-muted);",
            "        letter-spacing: 0.01rem;",
            "        text-transform: uppercase;",
            "    }",
            "    .result-card .result-stat .value {",
            "        font-size: 1.8rem;",
            "        font-weight: 700;",
            "        margin-top: 0.25rem;",
            "    }",
            "    .result-card pre {",
            "        margin-top: 1rem;",
            "        background: var(--surface-muted);",
            "        padding: 1rem;",
            "        border-radius: 1rem;",
            "        font-size: 0.85rem;",
            "        color: #121826;",
            "        overflow-x: auto;",
            "    }",
            "</style>",
        ]
    ),
    unsafe_allow_html=True,
)

def backend_alive():
    try:
        r = requests.get(f"{BACKEND}/health", timeout=3)
        return r.ok
    except Exception:
        return False

def render_ingest_result(result: dict):
    if not result:
        return

    status_text = str(result.get("status", "") or "").strip()
    normalized = status_text.lower()
    is_success = normalized in {"ok", "success", "done"} or not status_text
    pill_class = "ok" if is_success else "warn"
    pill_label = html.escape(status_text or "완료")

    pages_info = result.get("pages_index")
    if pages_info is None:
        pages_info = result.get("pages_indexed")

    detail_payload = None
    pages_value = None
    if isinstance(pages_info, (list, tuple, dict)):
        pages_value = len(pages_info)
        detail_payload = pages_info
    elif pages_info is not None:
        pages_value = pages_info

    stats_parts = []
    if pages_value is not None:
        stats_parts.append(
            "".join(
                [
                    '<div class="result-stat">',
                    '    <div class="label">인덱싱된 페이지</div>',
                    f'    <div class="value">{html.escape(str(pages_value))}</div>',
                    "</div>",
                ]
            )
        )

    excluded_keys = {"status", "pages_index", "pages_indexed"}
    for key, value in result.items():
        if key in excluded_keys:
            continue
        stats_parts.append(
            "".join(
                [
                    '<div class="result-stat">',
                    f'    <div class="label">{html.escape(str(key))}</div>',
                    f'    <div class="value">{html.escape(str(value))}</div>',
                    "</div>",
                ]
            )
        )

    detail_html = ""
    if isinstance(detail_payload, (dict, list)):
        pretty = json.dumps(detail_payload, ensure_ascii=False, indent=2)
        detail_html = f"<pre>{html.escape(pretty)}</pre>"

    stats_html = "".join(stats_parts) or (
        '<div class="result-stat">'
        '    <div class="label">결과</div>'
        '    <div class="value">-</div>'
        "</div>"
    )

    st.markdown(
        "".join(
            [
                '<div class="result-card">',
                f'    <div class="result-pill {pill_class}">{pill_label}</div>',
                f'    <div class="result-stats">{stats_html}</div>',
                f"    {detail_html}" if detail_html else "",
                "</div>",
            ]
        ),
        unsafe_allow_html=True,
    )

alive = backend_alive()

top_section = st.container()
bottom_section = st.container()

with top_section:
    st.markdown('<div class="top-flag"></div>', unsafe_allow_html=True)
    st.title("RAG — PDF · Image OCR + Multimodal Q&A")
    st.markdown(
        '<p class="page-lead">한 번의 업로드로 다양한 문서를 인덱싱하고, 텍스트·이미지를 혼합한 질문까지 즉시 처리하세요.</p>',
        unsafe_allow_html=True,
    )

    status_col1, status_col2 = st.columns([2, 1])
    with status_col1:
        st.markdown('<div class="backend-card-flag"></div>', unsafe_allow_html=True)
        st.markdown('<div class="backend-label">Backend URL</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="backend-url">{html.escape(BACKEND)}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="backend-helper">환경 변수 BACKEND_URL 기준 연결</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="backend-slider-label">검색 상위 K</div>', unsafe_allow_html=True)
        top_k = st.slider(
            "검색 상위 k",
            1,
            10,
            4,
            key="top_k_slider",
            label_visibility="collapsed",
        )
        st.markdown(
            f'<div class="slider-caption">질문 시 상위 {top_k}개 결과를 사용합니다.</div>',
            unsafe_allow_html=True,
        )

    status_state_class = "online" if alive else "offline"
    status_state_label = "연결됨" if alive else "중지됨"
    status_state_desc = "백엔드 헬스체크 성공" if alive else "응답 없음 — 컨테이너 확인"

    with status_col2:
        st.markdown(
            "".join(
                [
                    f'<div class="status-card highlight {status_state_class}">',
                    '    <div class="status-icon">',
                    '        <span class="status-dot"></span>',
                    "    </div>",
                    '    <div class="status-meta">',
                    '        <div class="label">Backend Health</div>',
                    f'        <div class="value">{status_state_label}</div>',
                    f'        <div class="status-pill">{status_state_desc}</div>',
                    "    </div>",
                    "</div>",
                ]
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        "\n".join(
            [
                '<div class="info-card">',
                '    <div class="chips">',
                '        <div class="chip">PDF</div>',
                '        <div class="chip">JPG · PNG · WEBP</div>',
                '        <div class="chip">멀티모달 질의</div>',
                '        <div class="chip">Top-K 검색</div>',
                "    </div>",
                "    <strong>워크플로 안내</strong>",
                '    <ol style="margin: 0.6rem 0 0; padding-left: 1.2rem; color: #1f2a37;">',
                "        <li>여러 문서를 묶어서 업로드 후 인덱싱합니다.</li>",
                "        <li>필요한 경우 이미지가 포함된 멀티모달 질문을 보냅니다.</li>",
                "        <li>요약, 근거 출처까지 한 화면에서 확인합니다.</li>",
                "    </ol>",
                "</div>",
            ]
        ),
        unsafe_allow_html=True,
    )

with bottom_section:
    st.markdown('<div class="bottom-flag"></div>', unsafe_allow_html=True)
    tab_ingest, tab_ask = st.tabs(["인덱싱 업로드", "질문하기"])

    with tab_ingest:
        st.markdown("#### 파일 업로드 & 인덱싱")
        st.write("PDF 및 이미지(jpg/png/webp)를 한 번에 여러 개 업로드할 수 있습니다.")

        ingest_files = st.file_uploader(
            "업로드할 파일 선택",
            type=["pdf", "jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="ingest_uploader",
        )

        col_ingest_btn, col_ingest_status = st.columns([1, 3])
        with col_ingest_btn:
            run_ingest = st.button("인덱싱 실행", disabled=(not ingest_files or not alive))
        with col_ingest_status:
            if run_ingest:
                if not ingest_files:
                    st.warning("파일을 업로드하세요.")
                else:
                    with st.spinner("백엔드로 업로드/인덱싱 중..."):
                        try:
                            multipart = []
                            for f in ingest_files:
                                multipart.append(("files", (f.name, f.getvalue(), f.type)))
                            resp = requests.post(f"{BACKEND}/ingest", files=multipart, timeout=600)
                            resp.raise_for_status()
                            result = resp.json()
                            st.success("인덱싱이 완료되었습니다.")
                            render_ingest_result(result if isinstance(result, dict) else {"status": "ok"})
                        except Exception as e:
                            st.error(f"인덱싱 실패: {e}")

    with tab_ask:
        st.markdown("#### 멀티모달 질문하기")
        q = st.text_input("질문을 입력하세요")
        q_imgs = st.file_uploader(
            "질문에 참고할 이미지 첨부(선택, 여러 장 가능)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="question_uploader",
        )

        if q_imgs:
            with st.expander("첨부 이미지 미리보기", expanded=False):
                cols = st.columns(4)
                for idx, img in enumerate(q_imgs):
                    with cols[idx % 4]:
                        st.image(img, caption=img.name, use_container_width=True)

        col_ask_btn, col_ask_status = st.columns([1, 3])
        with col_ask_btn:
            run_ask = st.button("질문 보내기", disabled=(not q or not alive))
        with col_ask_status:
            if run_ask:
                try:
                    if q_imgs:  # 멀티모달 질문 (/ask_mm)
                        form = {
                            "question": (None, q),
                            "top_k": (None, str(top_k)),
                        }
                        files_mp = []
                        for f in q_imgs:
                            files_mp.append(("images", (f.name, f.getvalue(), f.type)))
                        with st.spinner("질문(멀티모달) 처리 중..."):
                            resp = requests.post(f"{BACKEND}/ask_mm", files=files_mp, data=form, timeout=600)
                            resp.raise_for_status()
                            data = resp.json()
                    else:  # 텍스트 질문 (/ask)
                        with st.spinner("질문 처리 중..."):
                            resp = requests.post(
                                f"{BACKEND}/ask",
                                json={"question": q, "top_k": top_k, "max_new_tokens": 256},
                                timeout=600,
                            )
                            resp.raise_for_status()
                            data = resp.json()

                    st.subheader("답변")
                    st.write(data.get("answer", ""))

                    srcs = data.get("sources", [])
                    st.write("**참고한 출처:**", " · ".join(srcs) if srcs else "(없음)")

                except Exception as e:
                    st.error(f"질문 실패: {e}")

# ---- 푸터 ----
st.markdown("---")
st.caption("© RAG Demo — PDF, Image OCR & Multimodal Q&A")
