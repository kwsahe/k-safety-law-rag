const state = {
  user: null,
  conversations: [],
  conversationId: null,
  mode: window.location.pathname === "/general" ? "general" : "scenario",
  scenarioAnalysis: null,
  scenarioAnalysisEditing: false,
  csrfToken: "",
  registrationEnabled: false,
  lastPanelTrigger: null,
  lastSidebarTrigger: null,
};

const $ = (id) => document.getElementById(id);
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
let panelCloseTimer = null;

function currentTheme() {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

function applyTheme(theme, persist = true) {
  const nextTheme = theme === "dark" ? "dark" : "light";
  document.documentElement.dataset.theme = nextTheme;
  if (persist) localStorage.setItem("k-safety-theme", nextTheme);
  const button = $("theme-toggle");
  const icon = $("theme-toggle-icon");
  if (button) {
    const dark = nextTheme === "dark";
    button.setAttribute("aria-label", dark ? "라이트 모드로 전환" : "다크 모드로 전환");
    button.title = dark ? "라이트 모드로 전환" : "다크 모드로 전환";
    if (icon) icon.textContent = dark ? "☀" : "☾";
  }
}

function emptyStateHtml(mode = state.mode) {
  const isGeneral = mode === "general";
  const title = "채팅을 시작해보세요!";
  const modeTitle = isGeneral ? "일반 모드" : "시나리오 모드";
  const modeDescription = isGeneral
    ? "저장된 사고 시나리오 없이 산업안전보건법, 중대재해처벌법 등 일반 법령 질문에 답합니다."
    : "저장된 사고 시나리오를 함께 참고해 위반 여부, 책임 주체, 처벌 수위, 재발방지 조치를 판단합니다.";
  const examples = isGeneral
    ? ["안전보건교육과 특별안전교육의 차이를 설명해줘.", "산업안전보건법 제29조의 교육 의무를 정리해줘."]
    : ["비계 작업 특별안전교육 미실시가 위반인지 판단해줘.", "원청 책임을 산안법과 중대재해처벌법으로 나눠 설명해줘."];
  return `
    <div id="empty-state" class="mx-auto flex h-full max-w-3xl flex-col items-center justify-center px-4 text-center">
      <div class="mb-5 grid h-16 w-16 place-items-center rounded-3xl border border-blueLine/70 bg-navySoft text-2xl font-black text-navy shadow-sm">K</div>
      <h1 class="text-2xl font-black text-navyDeep">${title}</h1>
      <p class="mt-3 max-w-xl leading-7 text-mutedBlue">${modeDescription}</p>
      <div class="empty-mode-badge mt-5 inline-flex rounded-full border border-blueLine/70 bg-white/80 px-4 py-2 text-sm font-black text-navy">${modeTitle}</div>
      <div class="mt-6 grid w-full max-w-2xl gap-3 sm:grid-cols-2">
        ${examples.map((example) => `<button class="empty-example" type="button">${escapeHtml(example)}</button>`).join("")}
      </div>
    </div>
  `;
}

async function api(path, options = {}) {
  const method = String(options.method || "GET").toUpperCase();
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (state.csrfToken && ["POST", "PATCH", "DELETE"].includes(method) && !["/api/login", "/api/register"].includes(path)) {
    headers["X-CSRF-Token"] = state.csrfToken;
  }
  const response = await fetch(path, {
    ...options,
    headers,
    credentials: "same-origin",
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || "요청을 처리하지 못했습니다.");
  return data;
}

function showIntro() {
  $("intro").classList.remove("hidden");
  $("auth").classList.add("hidden");
  $("app").classList.add("hidden");
}

function showAuth(message = "") {
  $("intro").classList.add("hidden");
  $("auth").classList.remove("hidden");
  $("app").classList.add("hidden");
  setAuthMode("login");
  setAuthMessage(message, message ? "error" : "");
}

function showApp() {
  $("intro").classList.add("hidden");
  $("auth").classList.add("hidden");
  $("app").classList.remove("hidden");
  const roleLabel = state.user.role === "admin" ? "관리자" : "일반 사용자";
  $("user-badge").textContent = `${state.user.username} · ${roleLabel}`;
  $("admin-indicator").classList.toggle("hidden", state.user.role !== "admin");
  $("admin-dashboard-open").classList.toggle("hidden", state.user.role !== "admin");
  $("model-indicator").classList.toggle("hidden", state.user.role !== "admin");
  renderMode();
  if (state.user.role === "admin") refreshAdminHealth();
}

function startEntrance() {
  showIntro();
}

async function skipIntro() {
  await bootstrap();
}

function setAuthMode(mode) {
  if (mode === "register" && !state.registrationEnabled) mode = "login";
  const isLogin = mode === "login";
  $("login-form").classList.toggle("hidden", !isLogin);
  $("register-form").classList.toggle("hidden", isLogin);
  $("auth-login-tab").classList.toggle("auth-tab-active", isLogin);
  $("auth-register-tab").classList.toggle("auth-tab-active", !isLogin);
  $("auth-login-tab").setAttribute("aria-selected", String(isLogin));
  $("auth-register-tab").setAttribute("aria-selected", String(!isLogin));
}

function setAuthMessage(message = "", type = "") {
  const target = $("auth-message");
  target.textContent = message;
  target.className = `auth-message mt-5 text-sm font-semibold ${type ? `auth-message-${type}` : ""}`;
}

function syncModeUrl() {
  const target = state.mode === "general" ? "/general" : "/";
  if (window.location.pathname !== target) window.history.replaceState({}, "", target);
}

function modeLabel(mode = state.mode) {
  return mode === "general" ? "일반 법령" : "시나리오";
}

function renderMode() {
  const isGeneral = state.mode === "general";
  const isCompact = window.matchMedia("(max-width: 640px)").matches;
  $("mode-scenario").className = ["mode-button", isGeneral ? "" : "mode-button-active"].join(" ");
  $("mode-general").className = ["mode-button", isGeneral ? "mode-button-active" : ""].join(" ");
  $("mode-indicator").textContent = isGeneral ? "일반 법령" : isCompact ? "시나리오" : "시나리오 상담";
  $("scenario-open").disabled = isGeneral;
  $("scenario-open").classList.toggle("is-disabled", isGeneral);
  $("question").placeholder = isCompact
    ? "질문을 입력하세요"
    : isGeneral
      ? "법령 질문을 입력하세요"
      : "사고 시나리오를 바탕으로 질문하세요";
}

function setModelIndicator(status, label, title = label) {
  const indicator = $("model-indicator");
  indicator.className = `pill model-pill model-${status}`;
  indicator.innerHTML = '<span class="model-status-dot" aria-hidden="true"></span><span class="model-status-label"></span>';
  indicator.querySelector(".model-status-label").textContent = label;
  indicator.title = title;
  indicator.setAttribute("aria-label", title);
}

function renderConversations() {
  const list = $("conversation-list");
  list.innerHTML = "";

  const groups = [
    ["scenario", "시나리오 모드"],
    ["general", "일반 모드"],
  ];

  groups.forEach(([groupMode, groupTitle]) => {
    const conversations = state.conversations.filter((conv) => (conv.mode || "scenario") === groupMode);
    const section = document.createElement("section");
    section.className = "conversation-section";
    section.innerHTML = `<h3 class="conversation-section-title">${groupTitle}</h3>`;
    if (!conversations.length) {
      const empty = document.createElement("p");
      empty.className = "conversation-section-empty";
      empty.textContent = "아직 상담이 없습니다.";
      section.appendChild(empty);
      list.appendChild(section);
      return;
    }

    conversations.forEach((conv) => {
    const item = document.createElement("div");
    item.className = [
      "conversation-item group",
      conv.id === state.conversationId ? "conversation-item-active" : "",
    ].join(" ");
    item.innerHTML = `
      <button class="conversation-open" type="button">
        <span class="conversation-title">${escapeHtml(conv.title)}</span>
        <span class="conversation-mode">${modeLabel(conv.mode)}</span>
      </button>
      <button class="conversation-menu-button" type="button" aria-label="대화 메뉴">...</button>
      <div class="conversation-menu hidden">
        <button class="conversation-rename" type="button">이름 수정</button>
        <button class="conversation-delete" type="button">채팅 삭제</button>
      </div>
    `;

    item.querySelector(".conversation-open").onclick = () => {
      closeSidebarOnMobile();
      loadConversation(conv.id);
    };
    item.querySelector(".conversation-menu-button").onclick = (event) => {
      event.stopPropagation();
      toggleConversationMenu(item);
    };
    item.querySelector(".conversation-rename").onclick = (event) => {
      event.stopPropagation();
      closeConversationMenus();
      renameConversation(conv);
    };
    item.querySelector(".conversation-delete").onclick = (event) => {
      event.stopPropagation();
      closeConversationMenus();
      deleteConversation(conv);
    };

      section.appendChild(item);
    });
    list.appendChild(section);
  });
}

function closeConversationMenus() {
  document.querySelectorAll(".conversation-menu").forEach((menu) => menu.classList.add("hidden"));
}

function toggleConversationMenu(item) {
  const menu = item.querySelector(".conversation-menu");
  const isHidden = menu.classList.contains("hidden");
  closeConversationMenus();
  menu.classList.toggle("hidden", !isHidden);
}

function sourceLine(source, index) {
  const label = [source.law_name, source.article].filter(Boolean).join(" ");
  const rawPage = String(source.page || "").trim();
  const pageLabel = /^p\./i.test(rawPage) ? rawPage : `p.${rawPage}`;
  const page = rawPage && rawPage !== "페이지 정보 없음" ? ` · ${pageLabel}` : "";
  return `${index + 1}. ${label || "법령 근거"}${page}`;
}

function formatMessageTime(value) {
  const date = value ? new Date(value) : new Date();
  if (Number.isNaN(date.getTime())) return String(value || "");
  const parts = new Intl.DateTimeFormat("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).formatToParts(date);
  const map = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${map.year}.${map.month}.${map.day} ${map.hour}:${map.minute}:${map.second}`;
}

function setEmptyState() {
  const hasMessages = $("messages").querySelector(".message-row");
  $("empty-state")?.classList.toggle("hidden", Boolean(hasMessages));
}

function animateMessageSurface() {
  const surface = $("messages");
  surface.classList.remove("message-surface-enter");
  void surface.offsetWidth;
  surface.classList.add("message-surface-enter");
}

function appendSafeInlineMarkdown(target, text) {
  const parts = String(text).split(/(\*\*[^*]+\*\*)/g);
  parts.forEach((part) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      const strong = document.createElement("strong");
      strong.textContent = part.slice(2, -2);
      target.appendChild(strong);
    } else {
      target.appendChild(document.createTextNode(part));
    }
  });
}

function renderMessageContent(target, text, role) {
  target.textContent = "";
  if (role !== "assistant") {
    target.textContent = text;
    return;
  }

  String(text).split("\n").forEach((line) => {
    const row = document.createElement("div");
    row.className = "message-line";
    const plainLine = line.replace(/\*\*/g, "");
    const compact = plainLine.replace(/\s+/g, "");
    const isCritical = (
      /^(결론|위반\s*여부|처벌\s*수위|처벌\s*사항|과태료|법인\s*처벌)/.test(plainLine.trim())
      || /(\d+\s*년\s*(이상|이하).*(징역|금고)|\d+\s*억.*벌금|과태료.*\d)/.test(plainLine)
    );
    const isImportant = (
      /안전보건교육|안전\s*교육|특별안전교육|특별교육|재발방지|즉시\s*조치|작업중지/.test(plainLine)
      || /^\[(교육|재발방지|위반|처벌|책임|판단)/.test(plainLine.trim())
    );
    const isHeading = /^\[[^\]]+\]$/.test(plainLine.trim()) || /^\d+\.\s+\S/.test(plainLine.trim());
    if (isCritical) row.classList.add("message-line-critical");
    else if (isImportant) row.classList.add("message-line-important");
    if (isHeading) row.classList.add("message-line-heading");
    if (!compact) {
      row.appendChild(document.createTextNode("\u00a0"));
    } else {
      appendSafeInlineMarkdown(row, line);
    }
    target.appendChild(row);
  });
}

function messagesNearBottom() {
  const messages = $("messages");
  return messages.scrollHeight - messages.scrollTop - messages.clientHeight < 140;
}

function updateScrollBottomButton() {
  const button = $("scroll-bottom");
  const hasMessages = Boolean($("messages").querySelector(".message-row"));
  button.classList.toggle("hidden", !hasMessages || messagesNearBottom());
}

function scrollMessagesToBottom(smooth = false) {
  $("messages").scrollTo({
    top: $("messages").scrollHeight,
    behavior: smooth && !prefersReducedMotion.matches ? "smooth" : "auto",
  });
  window.setTimeout(updateScrollBottomButton, smooth ? 220 : 0);
}

function renderStatusContent(target, message) {
  const label = document.createElement("span");
  label.className = "thinking-label";
  label.textContent = message.content;
  target.appendChild(label);
  if (message.statusKind !== "complete") {
    const dots = document.createElement("span");
    dots.className = "thinking-dots";
    dots.setAttribute("aria-hidden", "true");
    for (let index = 0; index < 3; index += 1) {
      dots.appendChild(document.createElement("span"));
    }
    target.appendChild(dots);
  }
}

function appendMessage(message, options = {}) {
  const shouldScroll = Boolean(options.forceScroll) || messagesNearBottom();
  $("empty-state")?.classList.add("hidden");

  const row = document.createElement("article");
  if (message.id) row.dataset.messageId = String(message.id);
  row.className = [
    "message-row",
    message.role === "user" ? "message-row-user" : "message-row-assistant",
    message.status ? "message-row-status" : "",
    message.statusKind === "complete" ? "message-row-status-complete" : "",
    options.animate === false ? "message-no-animation" : "",
  ].join(" ");

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = message.role === "user" ? "나" : "K";

  const body = document.createElement("div");
  body.className = "message-body";

  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = `${message.role === "user" ? "입력 시간" : "출력 시간"} ${formatMessageTime(message.created_at)}`;
  body.appendChild(meta);

  const content = document.createElement("div");
  content.className = "message-content";
  if (message.status) renderStatusContent(content, message);
  else renderMessageContent(content, message.content, message.role);
  body.appendChild(content);

  if (!message.status) {
    const actions = document.createElement("div");
    actions.className = "message-actions";
    const copyButton = document.createElement("button");
    copyButton.className = "copy-button";
    copyButton.type = "button";
    copyButton.textContent = "복사";
    copyButton.onclick = () => copyText(message.content);
    actions.appendChild(copyButton);
    if (message.role === "assistant" && message.id) {
      actions.appendChild(createFeedbackButton(message, "helpful", "도움됨"));
      actions.appendChild(createFeedbackButton(message, "needs_improvement", "개선 필요"));
    }
    body.appendChild(actions);
  }

  if (message.role === "assistant" && message.payload) {
    const rawSources = Array.isArray(message.payload.sources) ? message.payload.sources : [];
    const sourceKeys = new Set();
    const sources = rawSources.filter((source) => {
      const key = [source.law_name, source.article, source.page].join("|");
      if (sourceKeys.has(key)) return false;
      sourceKeys.add(key);
      return true;
    });
    if (sources.length) {
      const details = document.createElement("details");
      details.className = "source-details";
      const summary = document.createElement("summary");
      summary.textContent = `법령 근거 ${sources.length}건`;
      details.appendChild(summary);
      const list = document.createElement("ol");
      list.className = "source-list";
      sources.forEach((source, index) => {
        const item = document.createElement("li");
        item.className = "source-item";
        item.textContent = sourceLine(source, index);
        list.appendChild(item);
      });
      details.appendChild(list);
      body.appendChild(details);
    }
    if (state.user.role === "admin" && message.payload.cli_output) {
      const details = document.createElement("details");
      details.className = "cli-details";
      const summary = document.createElement("summary");
      summary.textContent = "관리자 CLI 전체 출력";
      const toolbar = document.createElement("div");
      toolbar.className = "cli-toolbar";
      const copyCliButton = document.createElement("button");
      copyCliButton.className = "copy-button cli-copy-button";
      copyCliButton.type = "button";
      copyCliButton.textContent = "전체 복사";
      copyCliButton.setAttribute("aria-label", "관리자 CLI 전체 출력을 클립보드에 복사");
      copyCliButton.onclick = () => copyText(message.payload.cli_output);
      toolbar.appendChild(copyCliButton);
      const pre = document.createElement("pre");
      pre.className = "cli-output";
      pre.textContent = message.payload.cli_output;
      details.appendChild(summary);
      details.appendChild(toolbar);
      details.appendChild(pre);
      body.appendChild(details);
    }
  }

  row.appendChild(avatar);
  row.appendChild(body);
  $("messages").appendChild(row);
  if (shouldScroll) scrollMessagesToBottom(Boolean(options.smooth));
  else updateScrollBottomButton();
  return row;
}

function hydratePendingUserMessage(row, message) {
  if (!row || !message) return;
  if (message.id) row.dataset.messageId = String(message.id);
  const meta = row.querySelector(".message-meta");
  if (meta) meta.textContent = `입력 시간 ${formatMessageTime(message.created_at)}`;
}

function createFeedbackButton(message, rating, label) {
  const button = document.createElement("button");
  button.className = "feedback-button";
  button.type = "button";
  button.textContent = label;
  button.dataset.rating = rating;
  button.classList.toggle("feedback-button-active", message.feedback?.rating === rating);
  button.onclick = async () => {
    let comment = message.feedback?.comment || "";
    if (rating === "needs_improvement" && window.Swal) {
      const result = await Swal.fire({
        title: "어떤 점을 개선할까요?",
        input: "textarea",
        inputValue: comment,
        inputPlaceholder: "누락된 조항이나 잘못된 판단을 적어주세요.",
        inputAttributes: { maxlength: 500 },
        showCancelButton: true,
        confirmButtonText: "평가 저장",
        cancelButtonText: "취소",
        confirmButtonColor: "#233f73",
        background: "#ffffff",
        color: "#1d2935",
      });
      if (!result.isConfirmed) return;
      comment = String(result.value || "").trim();
    }
    try {
      const data = await api(`/api/messages/${message.id}/feedback`, {
        method: "POST",
        body: JSON.stringify({ rating, comment }),
      });
      message.feedback = data.feedback;
      button.parentElement.querySelectorAll(".feedback-button").forEach((item) => {
        item.classList.toggle("feedback-button-active", item.dataset.rating === rating);
      });
    } catch (error) {
      showError("평가 저장 실패", error.message);
    }
  };
  return button;
}

function formatMetric(value) {
  return new Intl.NumberFormat("ko-KR").format(Number(value || 0));
}

function intentLabel(intent) {
  const labels = {
    comprehensive_report: "종합 보고서",
    prime_contractor_liability: "원청·도급 책임",
    executive_liability: "경영책임자 책임",
    employer_liability: "사업주 책임",
    ppe_scaffold_standards: "보호구·비계 기준",
    scaffold_special_education: "비계 특별교육",
    education_comparison: "교육 제도 비교",
    scenario_general: "시나리오 일반",
    general_law: "일반 법령",
    unknown: "이전 답변",
  };
  return labels[intent] || intent;
}

async function loadAdminDashboard() {
  const data = await api("/api/admin/dashboard");
  const { totals, quality, feedback } = data;
  $("admin-metrics").innerHTML = `
    <article class="metric-card"><span>사용자</span><strong>${formatMetric(totals.users)}</strong></article>
    <article class="metric-card"><span>활성 상담</span><strong>${formatMetric(totals.conversations)}</strong></article>
    <article class="metric-card"><span>저장 답변</span><strong>${formatMetric(totals.answers)}</strong></article>
    <article class="metric-card"><span>평균 응답</span><strong>${quality.average_elapsed_ms ? `${(quality.average_elapsed_ms / 1000).toFixed(1)}초` : "-"}</strong></article>
  `;

  const citation = quality.citation || {};
  const checked = Number(citation.pass || 0) + Number(citation.warn || 0) + Number(citation.fail || 0);
  const passRate = checked ? `${Math.round((Number(citation.pass || 0) / checked) * 100)}%` : "-";
  $("citation-summary").innerHTML = `
    <div class="quality-score"><strong>${passRate}</strong><span>검증 통과율</span></div>
    <div class="quality-status quality-pass"><span>통과</span><strong>${formatMetric(citation.pass)}</strong></div>
    <div class="quality-status quality-warn"><span>확인</span><strong>${formatMetric(citation.warn)}</strong></div>
    <div class="quality-status quality-fail"><span>실패</span><strong>${formatMetric(citation.fail)}</strong></div>
  `;

  const intents = Object.entries(quality.intent_counts || {}).sort((a, b) => b[1] - a[1]);
  $("intent-summary").innerHTML = intents.length
    ? intents.map(([intent, count]) => `<div class="intent-row"><span>${escapeHtml(intentLabel(intent))}</span><strong>${formatMetric(count)}</strong></div>`).join("")
    : '<p class="dashboard-empty">아직 분류된 답변이 없습니다.</p>';

  const feedbackCounts = feedback.counts || {};
  const feedbackHeader = `<div class="feedback-counts"><span>도움됨 <strong>${formatMetric(feedbackCounts.helpful)}</strong></span><span>개선 필요 <strong>${formatMetric(feedbackCounts.needs_improvement)}</strong></span></div>`;
  const recent = feedback.recent || [];
  $("feedback-summary").innerHTML = feedbackHeader + (recent.length
    ? recent.map((item) => `
        <article class="feedback-entry">
          <div><strong>${item.rating === "helpful" ? "도움됨" : "개선 필요"}</strong><span>${escapeHtml(item.username)} · ${escapeHtml(item.title)}</span></div>
          <p>${escapeHtml(item.comment || "별도 의견 없음")}</p>
        </article>
      `).join("")
    : '<p class="dashboard-empty">아직 저장된 사용자 평가가 없습니다.</p>');
}

async function refreshAdminHealth() {
  if (state.user?.role !== "admin") return;
  const banner = $("admin-health");
  setModelIndicator("checking", "확인 중", "EXAONE 연결 상태 확인 중");
  if (banner) banner.textContent = "EXAONE 모델과 데이터 저장소의 상태를 확인하고 있습니다.";
  try {
    const health = await api("/api/admin/health");
    const connected = Boolean(health.model.connected);
    setModelIndicator(
      connected ? "online" : "offline",
      connected ? "정상" : "확인 필요",
      connected ? "EXAONE 연결 정상" : `EXAONE 연결 확인 필요: ${health.model.detail}`,
    );
    if (banner) {
      banner.className = `health-banner ${connected ? "health-online" : "health-warning"}`;
      banner.innerHTML = `
        <div><strong>${connected ? "시스템 정상" : "모델 연결 확인 필요"}</strong><span>${escapeHtml(health.model.detail)}</span></div>
        <dl><div><dt>모델</dt><dd>${escapeHtml(health.model.name)}</dd></div><div><dt>응답</dt><dd>${formatMetric(health.model.latency_ms)}ms</dd></div><div><dt>DB</dt><dd>${escapeHtml(health.database)}</dd></div><div><dt>Vector DB</dt><dd>${escapeHtml(health.vector_db)}</dd></div></dl>
      `;
    }
  } catch (error) {
    setModelIndicator("offline", "확인 실패", `모델 상태 확인 실패: ${error.message}`);
    if (banner) {
      banner.className = "health-banner health-warning";
      banner.textContent = error.message;
    }
  }
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function copyText(text) {
  let copied = false;
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      copied = true;
    } catch {
      copied = false;
    }
  }

  if (!copied) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    textarea.setSelectionRange(0, textarea.value.length);
    copied = document.execCommand("copy");
    textarea.remove();
  }

  if (window.Swal) {
    await Swal.fire({
      icon: copied ? "success" : "error",
      title: copied ? "복사 완료" : "복사 실패",
      text: copied ? "클립보드에 복사했습니다." : "브라우저의 클립보드 권한을 확인해주세요.",
      timer: copied ? 900 : undefined,
      showConfirmButton: !copied,
      confirmButtonColor: "#233f73",
      background: "#ffffff",
      color: "#1d2935",
    });
  }
  return copied;
}

async function renameConversation(conv) {
  const result = window.Swal
    ? await Swal.fire({
        title: "상담 이름 수정",
        input: "text",
        inputValue: conv.title,
        inputAttributes: { maxlength: 80 },
        showCancelButton: true,
        confirmButtonText: "저장",
        cancelButtonText: "취소",
        confirmButtonColor: "#233f73",
        background: "#ffffff",
        color: "#1d2935",
      })
    : { isConfirmed: true, value: prompt("상담 이름", conv.title) };
  if (!result.isConfirmed) return;
  const title = String(result.value || "").trim();
  if (!title) return;
  try {
    const data = await api(`/api/conversations/${conv.id}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    });
    state.conversations = state.conversations.map((item) => (item.id === conv.id ? data.conversation : item));
    if (state.conversationId === conv.id) $("chat-title").textContent = data.conversation.title;
    renderConversations();
  } catch (error) {
    showError("이름 수정 실패", error.message);
  }
}

async function deleteConversation(conv) {
  const result = window.Swal
    ? await Swal.fire({
        icon: "warning",
        title: "채팅 삭제",
        text: `"${conv.title}" 상담을 화면에서 삭제합니다. DB에는 삭제 로그가 남습니다.`,
        showCancelButton: true,
        confirmButtonText: "삭제",
        cancelButtonText: "취소",
        confirmButtonColor: "#ff4e72",
        background: "#ffffff",
        color: "#1d2935",
      })
    : { isConfirmed: confirm(`"${conv.title}" 채팅을 삭제할까요? DB에는 삭제 로그가 남습니다.`) };
  if (!result.isConfirmed) return;
  try {
    await api(`/api/conversations/${conv.id}`, { method: "DELETE" });
    const wasActive = state.conversationId === conv.id;
    await refreshConversations();
    if (wasActive) {
      const next = state.conversations.find((item) => (item.mode || "scenario") === state.mode) || state.conversations[0];
      if (next) await loadConversation(next.id);
      else startDraft();
    }
  } catch (error) {
    showError("삭제 실패", error.message);
  }
}

async function showError(title, message) {
  if (window.Swal) {
    await Swal.fire({
      icon: "error",
      title,
      text: message,
      confirmButtonText: "확인",
      confirmButtonColor: "#233f73",
      background: "#ffffff",
      color: "#1d2935",
    });
  } else {
    alert(`${title}\n${message}`);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function refreshConversations() {
  const data = await api("/api/conversations");
  state.conversations = data.conversations;
  renderConversations();
}

function startDraft() {
  const title = state.mode === "general" ? "새 일반 법령 상담" : "새 시나리오 상담";
  state.conversationId = null;
  $("chat-title").textContent = title;
  renderEmptyState();
  renderMode();
  renderConversations();
  syncModeUrl();
}

async function loadConversation(id) {
  const data = await api(`/api/conversations/${id}`);
  state.conversationId = data.conversation.id;
  state.mode = data.conversation.mode || "scenario";
  syncModeUrl();
  $("chat-title").textContent = data.conversation.title;
  $("messages").innerHTML = "";
  data.messages.forEach((message) => appendMessage(message, { animate: false }));
  if (!data.messages.length) {
    renderEmptyState();
  }
  renderConversations();
  renderMode();
  setEmptyState();
  animateMessageSurface();
  requestAnimationFrame(() => scrollMessagesToBottom(false));
}

function renderEmptyState() {
  $("messages").innerHTML = emptyStateHtml(state.mode);
  animateMessageSurface();
  $("messages").querySelectorAll(".empty-example").forEach((button) => {
    button.addEventListener("click", () => {
      $("question").value = button.textContent.trim();
      $("question").focus();
      $("question").dispatchEvent(new Event("input"));
    });
  });
}

async function loadScenario() {
  const data = await api("/api/scenario");
  $("scenario-overview").value = data.scenario.overview || "";
  $("scenario-details").value = data.scenario.details || "";
  $("scenario-workers").value = data.scenario.workers || "";
  state.scenarioAnalysis = data.analysis || null;
  state.scenarioAnalysisEditing = false;
  renderScenarioAnalysis();
}

function analysisListValue(value) {
  return Array.isArray(value) ? value.join("\n") : "";
}

function analysisListFromInput(value) {
  return String(value || "")
    .split(/[\n,]+/)
    .map((item) => item.trim())
    .filter((item, index, items) => item && items.indexOf(item) === index);
}

function renderScenarioAnalysisEditor(profile) {
  const content = $("scenario-analysis-content");
  const contracts = ["직영", "도급", "파견", "도급/파견 혼재", "확인 필요"];
  content.innerHTML = `
    <form id="scenario-analysis-form" class="scenario-analysis-form">
      <p class="scenario-analysis-notice">LLM이 잘못 읽은 사고 사실을 교정하세요. 법령과 처벌 기준은 답변 생성 단계에서 별도로 검증됩니다.</p>
      <div class="scenario-analysis-editor-grid">
        <label>사고 유형<input name="accident_type" maxlength="40" value="${escapeHtml(profile.accident_type || "기타")}" required></label>
        <label>계약 구조
          <select name="contract_structure">
            ${contracts.map((item) => `<option value="${escapeHtml(item)}" ${profile.contract_structure === item ? "selected" : ""}>${escapeHtml(item)}</option>`).join("")}
          </select>
        </label>
        <label>사업주ㆍ원청<input name="company" maxlength="120" value="${escapeHtml(profile.company || "확인 필요")}"></label>
        <label>수급ㆍ협력업체<input name="contractor" maxlength="120" value="${escapeHtml(profile.contractor || "확인 필요")}"></label>
        <label>사망자 수<input name="death_count" type="number" min="0" max="10000" value="${Number(profile.death_count || 0)}"></label>
        <label>부상자 수<input name="injury_count" type="number" min="0" max="10000" value="${Number(profile.injury_count || 0)}"></label>
        <label>6개월 이상 부상자 수<input name="long_term_injury_count" type="number" min="0" max="10000" value="${profile.long_term_injury_count == null ? "" : Number(profile.long_term_injury_count)}" placeholder="확인되지 않으면 비워두세요"></label>
        <label class="scenario-analysis-editor-wide">작업 종류<textarea name="work_types" rows="2" placeholder="한 줄에 하나씩 입력">${escapeHtml(analysisListValue(profile.work_types))}</textarea></label>
        <label class="scenario-analysis-editor-wide">위험요인<textarea name="hazards" rows="3" placeholder="한 줄에 하나씩 입력">${escapeHtml(analysisListValue(profile.hazards))}</textarea></label>
        <label class="scenario-analysis-editor-wide">미조치 사항<textarea name="missing_controls" rows="3" placeholder="한 줄에 하나씩 입력">${escapeHtml(analysisListValue(profile.missing_controls))}</textarea></label>
        <label class="scenario-analysis-editor-wide">추가 확인 사항<textarea name="uncertainties" rows="2" placeholder="한 줄에 하나씩 입력">${escapeHtml(analysisListValue(profile.uncertainties))}</textarea></label>
      </div>
      <div class="scenario-analysis-form-actions">
        <button id="scenario-analysis-cancel-edit" class="secondary-button" type="button">취소</button>
        <button id="scenario-analysis-save-edit" class="primary-button" type="submit">수정 저장</button>
      </div>
    </form>
  `;

  $("scenario-analysis-cancel-edit").addEventListener("click", () => {
    state.scenarioAnalysisEditing = false;
    renderScenarioAnalysis();
  });
  $("scenario-analysis-form").addEventListener("submit", saveScenarioAnalysisEdit);
}

async function saveScenarioAnalysisEdit(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const submit = $("scenario-analysis-save-edit");
  const formData = new FormData(form);
  submit.disabled = true;
  submit.textContent = "저장 중...";
  try {
    const profile = {
      accident_type: formData.get("accident_type"),
      work_types: analysisListFromInput(formData.get("work_types")),
      company: formData.get("company"),
      contractor: formData.get("contractor"),
      contract_structure: formData.get("contract_structure"),
      death_count: formData.get("death_count"),
      injury_count: formData.get("injury_count"),
      long_term_injury_count: formData.get("long_term_injury_count"),
      hazards: analysisListFromInput(formData.get("hazards")),
      missing_controls: analysisListFromInput(formData.get("missing_controls")),
      uncertainties: analysisListFromInput(formData.get("uncertainties")),
    };
    const data = await api("/api/scenario/analysis", {
      method: "PATCH",
      body: JSON.stringify({ profile }),
    });
    state.scenarioAnalysis = data.analysis;
    state.scenarioAnalysisEditing = false;
    renderScenarioAnalysis();
    $("scenario-message").textContent = "분석 내용을 수정했습니다. 이후 질문부터 교정된 사실을 적용합니다.";
  } catch (error) {
    $("scenario-message").textContent = error.message;
    showError("분석 수정 실패", error.message);
    submit.disabled = false;
    submit.textContent = "수정 저장";
  }
}

function renderScenarioAnalysis() {
  const analysis = state.scenarioAnalysis || { status: "not_analyzed", profile: null };
  const status = analysis.status || "not_analyzed";
  const badge = $("scenario-analysis-status");
  const content = $("scenario-analysis-content");
  const editButton = $("scenario-analysis-edit");
  const labels = {
    not_analyzed: "분석 전",
    pending: "분석 필요",
    analyzing: "분석 중",
    complete: "분석 완료",
    failed: "분석 실패",
  };
  const userEdited = Boolean(analysis.profile?.validation?.user_edited);
  badge.textContent = userEdited ? "사용자 수정 완료" : (labels[status] || "분석 전");
  badge.className = "scenario-analysis-status";
  if (status === "analyzing") badge.classList.add("is-running");
  if (status === "complete") badge.classList.add("is-complete");
  if (status === "failed") badge.classList.add("is-failed");
  editButton.classList.toggle("hidden", status !== "complete" || !analysis.profile || state.scenarioAnalysisEditing);

  if (status === "complete" && analysis.profile) {
    const profile = analysis.profile;
    if (state.scenarioAnalysisEditing) {
      renderScenarioAnalysisEditor(profile);
      return;
    }
    const workTypes = (profile.work_types || []).join(", ") || "확인 필요";
    const hazards = (profile.hazards || []).join(", ") || "확인 필요";
    const controls = (profile.missing_controls || []).join(", ") || "명시된 내용 없음";
    const uncertainties = (profile.uncertainties || []).join(", ") || "없음";
    const longTermCount = profile.long_term_injury_count == null ? "확인 필요" : `${Number(profile.long_term_injury_count)}명`;
    content.innerHTML = `
      <div class="scenario-analysis-grid">
        <div class="scenario-analysis-item"><span>사고 유형</span><strong>${escapeHtml(profile.accident_type || "기타")}</strong></div>
        <div class="scenario-analysis-item"><span>작업</span><strong>${escapeHtml(workTypes)}</strong></div>
        <div class="scenario-analysis-item"><span>사업주ㆍ원청</span><strong>${escapeHtml(profile.company || "확인 필요")}</strong></div>
        <div class="scenario-analysis-item"><span>수급ㆍ협력업체</span><strong>${escapeHtml(profile.contractor || "확인 필요")}</strong></div>
        <div class="scenario-analysis-item"><span>계약 구조</span><strong>${escapeHtml(profile.contract_structure || "확인 필요")}</strong></div>
        <div class="scenario-analysis-item"><span>사망</span><strong>${Number(profile.death_count || 0)}명</strong></div>
        <div class="scenario-analysis-item"><span>부상</span><strong>${Number(profile.injury_count || 0)}명</strong></div>
        <div class="scenario-analysis-item"><span>6개월 이상 부상</span><strong>${escapeHtml(longTermCount)}</strong></div>
        <div class="scenario-analysis-item scenario-analysis-wide"><span>위험요인</span><strong>${escapeHtml(hazards)}</strong></div>
        <div class="scenario-analysis-item scenario-analysis-wide"><span>미조치 사항</span><strong>${escapeHtml(controls)}</strong></div>
        <div class="scenario-analysis-item scenario-analysis-wide"><span>추가 확인 사항</span><strong>${escapeHtml(uncertainties)}</strong></div>
      </div>
    `;
    return;
  }
  if (status === "failed") {
    content.textContent = analysis.error || "모델 연결과 시나리오 내용을 확인한 뒤 다시 실행하세요.";
    return;
  }
  if (status === "analyzing") {
    content.textContent = "EXAONE이 사고 사실을 구조화하고 있습니다. 잠시만 기다려 주세요.";
    return;
  }
  content.textContent = status === "pending"
    ? "저장된 내용이 변경되었습니다. LLM 분석을 다시 실행하세요."
    : "저장 후 LLM 분석을 실행하면 사고 사실을 구조화합니다.";
}

async function saveScenario() {
  const data = await api("/api/scenario", {
    method: "POST",
    body: JSON.stringify({
      overview: $("scenario-overview").value,
      details: $("scenario-details").value,
      workers: $("scenario-workers").value,
    }),
  });
  state.scenarioAnalysis = data.analysis || null;
  state.scenarioAnalysisEditing = false;
  renderScenarioAnalysis();
  return data;
}

async function bootstrap() {
  try {
    const data = await api("/api/me");
    state.user = data.user;
    state.csrfToken = data.csrf_token || "";
    state.registrationEnabled = Boolean(data.registration_enabled);
    $("auth-register-tab").classList.toggle("hidden", !state.registrationEnabled);
    if (!state.user) {
      showAuth();
      return;
    }
    showApp();
    await refreshConversations();
    const preferred = state.conversations.find((conv) => (conv.mode || "scenario") === state.mode);
    if (preferred) await loadConversation(preferred.id);
    else startDraft();
    await loadScenario();
  } catch (error) {
    showAuth(error.message);
  }
}

$("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const data = await api("/api/login", {
      method: "POST",
      body: JSON.stringify({
        username: $("login-username").value,
        password: $("login-password").value,
      }),
    });
    state.user = data.user;
    state.csrfToken = data.csrf_token || "";
    await bootstrap();
  } catch (error) {
    showAuth(error.message);
  }
});

$("intro-skip").addEventListener("click", skipIntro);

$("auth-login-tab").addEventListener("click", () => {
  setAuthMode("login");
  setAuthMessage();
  $("login-username").focus();
});

$("auth-register-tab").addEventListener("click", () => {
  setAuthMode("register");
  setAuthMessage();
  $("register-username").focus();
});

$("register-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api("/api/register", {
      method: "POST",
      body: JSON.stringify({
        username: $("register-username").value,
        password: $("register-password").value,
      }),
    });
    setAuthMode("login");
    $("login-username").value = $("register-username").value.trim();
    $("register-password").value = "";
    setAuthMessage("일반 계정이 생성되었습니다. 로그인하세요.", "success");
    $("login-password").focus();
  } catch (error) {
    setAuthMessage(error.message, "error");
  }
});

$("logout").addEventListener("click", async () => {
  await api("/api/logout", { method: "POST", body: "{}" });
  state.user = null;
  state.csrfToken = "";
  state.conversations = [];
  state.conversationId = null;
  showAuth();
});

$("new-chat").addEventListener("click", async () => {
  closeSidebarOnMobile();
  startDraft();
});

document.addEventListener("click", (event) => {
  if (!event.target.closest(".conversation-menu") && !event.target.closest(".conversation-menu-button")) {
    closeConversationMenus();
  }
});

$("mode-scenario").addEventListener("click", async () => {
  if (state.mode === "scenario") return;
  state.mode = "scenario";
  closeSidebarOnMobile();
  startDraft();
});

$("mode-general").addEventListener("click", async () => {
  if (state.mode === "general") return;
  state.mode = "general";
  closePanels();
  closeSidebarOnMobile();
  startDraft();
});

$("sidebar-toggle").addEventListener("click", () => {
  const willOpen = !$("sidebar").classList.contains("sidebar-open");
  if (willOpen) state.lastSidebarTrigger = document.activeElement;
  $("sidebar").classList.toggle("sidebar-open", willOpen);
  $("sidebar-backdrop").classList.toggle("hidden", !willOpen);
  if (willOpen) window.setTimeout(() => focusableElements($("sidebar"))[0]?.focus(), 80);
});

function closeSidebarOnMobile(restoreFocus = true) {
  const wasOpen = $("sidebar").classList.contains("sidebar-open");
  $("sidebar").classList.remove("sidebar-open");
  $("sidebar-backdrop").classList.add("hidden");
  if (wasOpen && restoreFocus && state.lastSidebarTrigger instanceof HTMLElement) state.lastSidebarTrigger.focus();
  state.lastSidebarTrigger = null;
}

function focusableElements(container) {
  return Array.from(container.querySelectorAll(
    'button:not([disabled]), [href], input:not([disabled]), textarea:not([disabled]), select:not([disabled]), summary, [tabindex]:not([tabindex="-1"])',
  )).filter((element) => !element.classList.contains("hidden") && element.getClientRects().length);
}

function activePanel() {
  return [$("scenario-panel"), $("admin-dashboard")].find((panel) => !panel.classList.contains("hidden")) || null;
}

function openPanel(panelId, returnFocus = document.activeElement) {
  closePanels(true, false);
  window.clearTimeout(panelCloseTimer);
  const panel = $(panelId);
  state.lastPanelTrigger = returnFocus;
  panel.classList.remove("hidden", "panel-closing");
  $("panel-backdrop").classList.remove("hidden");
  document.body.classList.add("panel-open");
  requestAnimationFrame(() => panel.classList.add("panel-visible"));
  if (panelId === "scenario-panel" && window.matchMedia("(max-width: 640px)").matches) {
    panel.querySelectorAll(".scenario-input-section").forEach((section, index) => {
      section.open = index === 0;
    });
  }
  window.setTimeout(() => focusableElements(panel)[0]?.focus(), 60);
}

function closePanels(immediate = false, restoreFocus = true) {
  window.clearTimeout(panelCloseTimer);
  const panels = [$("scenario-panel"), $("admin-dashboard")];
  const visiblePanels = panels.filter((panel) => !panel.classList.contains("hidden"));
  if (!visiblePanels.length) return;
  const finish = () => {
    panels.forEach((panel) => panel.classList.add("hidden"));
    panels.forEach((panel) => panel.classList.remove("panel-visible", "panel-closing"));
    $("panel-backdrop").classList.add("hidden");
    $("panel-backdrop").classList.remove("backdrop-closing");
    document.body.classList.remove("panel-open");
    if (restoreFocus && state.lastPanelTrigger instanceof HTMLElement) state.lastPanelTrigger.focus();
    state.lastPanelTrigger = null;
  };
  if (immediate || prefersReducedMotion.matches) {
    finish();
    return;
  }
  visiblePanels.forEach((panel) => panel.classList.add("panel-closing"));
  $("panel-backdrop").classList.add("backdrop-closing");
  panelCloseTimer = window.setTimeout(finish, 170);
}

$("scenario-open").addEventListener("click", async () => {
  if (state.mode === "general") return;
  const returnFocus = window.innerWidth < 1024 ? $("sidebar-toggle") : $("scenario-open");
  closeSidebarOnMobile(false);
  await loadScenario();
  openPanel("scenario-panel", returnFocus);
});

$("scenario-close").addEventListener("click", () => closePanels());

$("admin-dashboard-open").addEventListener("click", async () => {
  const returnFocus = window.innerWidth < 1024 ? $("sidebar-toggle") : $("admin-dashboard-open");
  closeSidebarOnMobile(false);
  openPanel("admin-dashboard", returnFocus);
  try {
    await Promise.all([loadAdminDashboard(), refreshAdminHealth()]);
  } catch (error) {
    showError("품질 현황 불러오기 실패", error.message);
  }
});

$("admin-dashboard-close").addEventListener("click", () => closePanels());
$("panel-backdrop").addEventListener("click", () => closePanels());
$("sidebar-backdrop").addEventListener("click", () => closeSidebarOnMobile());

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeConversationMenus();
    closePanels();
    closeSidebarOnMobile();
    return;
  }
  if (event.key !== "Tab") return;
  const container = activePanel()
    || (window.innerWidth < 1024 && $("sidebar").classList.contains("sidebar-open") ? $("sidebar") : null);
  if (!container) return;
  const elements = focusableElements(container);
  if (!elements.length) return;
  const first = elements[0];
  const last = elements[elements.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
});

$("admin-health-refresh").addEventListener("click", refreshAdminHealth);

$("scenario-analysis-edit").addEventListener("click", () => {
  state.scenarioAnalysisEditing = true;
  renderScenarioAnalysis();
});

$("scenario-save").addEventListener("click", async () => {
  const button = $("scenario-save");
  button.disabled = true;
  button.textContent = "저장 중...";
  try {
    await saveScenario();
    $("scenario-message").textContent = "저장했습니다.";
    button.textContent = "✓ 저장 완료";
    await wait(700);
  } catch (error) {
    $("scenario-message").textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = "시나리오 저장";
  }
});

$("scenario-analyze").addEventListener("click", async () => {
  const button = $("scenario-analyze");
  const saveButton = $("scenario-save");
  button.disabled = true;
  saveButton.disabled = true;
  button.classList.add("is-loading");
  button.textContent = "분석 중...";
  $("scenario-message").textContent = "현재 내용을 저장하고 EXAONE 분석을 시작합니다.";
  state.scenarioAnalysis = { status: "analyzing", profile: null };
  state.scenarioAnalysisEditing = false;
  renderScenarioAnalysis();
  try {
    await saveScenario();
    state.scenarioAnalysis = { status: "analyzing", profile: null };
    renderScenarioAnalysis();
    const data = await api("/api/scenario/analyze", { method: "POST", body: "{}" });
    state.scenarioAnalysis = data.analysis;
    renderScenarioAnalysis();
    const seconds = Math.max(0.1, Number(data.elapsed_ms || 0) / 1000).toFixed(1);
    $("scenario-message").textContent = `분석을 완료했습니다. 이후 질문에 적용됩니다. (${seconds}초)`;
  } catch (error) {
    await loadScenario().catch(() => {});
    $("scenario-message").textContent = error.message;
    showError("시나리오 분석 실패", error.message);
  } finally {
    button.disabled = false;
    saveButton.disabled = false;
    button.classList.remove("is-loading");
    button.textContent = "LLM 분석 시작";
  }
});

$("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;
  $("question").value = "";
  $("question").style.height = "auto";

  const pendingMessage = appendMessage(
    { role: "user", content: question, created_at: new Date().toISOString() },
    { forceScroll: true, smooth: true },
  );
  let statusMessage = appendMessage(
    { role: "assistant", content: "생각 중", created_at: new Date().toISOString(), status: true },
    { forceScroll: true, smooth: true },
  );
  statusMessage.setAttribute("role", "status");
  statusMessage.setAttribute("aria-live", "polite");
  $("send").disabled = true;
  $("send").textContent = "생성 중";

  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ conversation_id: state.conversationId, question, mode: state.mode }),
    });
    state.conversationId = data.conversation_id;
    state.mode = data.mode || state.mode;
    statusMessage.remove();
    hydratePendingUserMessage(pendingMessage, data.user_message);
    statusMessage = appendMessage(
      { role: "assistant", content: "정리 완료!", created_at: new Date().toISOString(), status: true, statusKind: "complete" },
      { forceScroll: true },
    );
    await wait(650);
    statusMessage.remove();
    appendMessage(data.message, { forceScroll: true, smooth: true });
    await refreshConversations();
    const activeConversation = state.conversations.find((item) => item.id === state.conversationId);
    if (activeConversation) $("chat-title").textContent = activeConversation.title;
    renderMode();
    syncModeUrl();
  } catch (error) {
    statusMessage.remove();
    const message = error.message || "응답을 생성하지 못했습니다.";
    const isModelError = /모델|EXAONE|Colab|LLM|memory|연결/i.test(message);
    const errorTitle = isModelError ? "모델 연결 확인 필요" : "응답 생성 실패";
    if (window.Swal) {
      await Swal.fire({
        icon: "error",
        title: errorTitle,
        text: message,
        confirmButtonText: "확인",
        confirmButtonColor: "#233f73",
        background: "#ffffff",
        color: "#1d2935",
      });
    }
    appendMessage({ role: "assistant", content: `${errorTitle}: ${message}` });
  } finally {
    $("send").disabled = false;
    $("send").textContent = "전송";
  }
});

$("question").addEventListener("input", (event) => {
  event.target.style.height = "auto";
  event.target.style.height = `${Math.min(event.target.scrollHeight, 180)}px`;
});

$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("chat-form").requestSubmit();
  }
});

$("messages").addEventListener("scroll", updateScrollBottomButton, { passive: true });
$("scroll-bottom").addEventListener("click", () => scrollMessagesToBottom(true));

window.addEventListener("resize", renderMode);

$("theme-toggle").addEventListener("click", () => {
  applyTheme(currentTheme() === "dark" ? "light" : "dark");
});

applyTheme(currentTheme(), false);
startEntrance();
