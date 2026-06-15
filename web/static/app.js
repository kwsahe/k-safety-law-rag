const state = {
  user: null,
  conversations: [],
  conversationId: null,
  mode: window.location.pathname === "/general" ? "general" : "scenario",
};

const $ = (id) => document.getElementById(id);

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
      <div class="mt-5 inline-flex rounded-full border border-blueLine/70 bg-white/80 px-4 py-2 text-sm font-black text-navy">${modeTitle}</div>
      <div class="mt-6 grid w-full max-w-2xl gap-3 sm:grid-cols-2">
        ${examples.map((example) => `<button class="empty-example" type="button">${escapeHtml(example)}</button>`).join("")}
      </div>
    </div>
  `;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    credentials: "same-origin",
    ...options,
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
  $("auth-message").textContent = message;
}

function showApp() {
  $("intro").classList.add("hidden");
  $("auth").classList.add("hidden");
  $("app").classList.remove("hidden");
  $("user-badge").textContent = `${state.user.username} · ${state.user.role}`;
  $("admin-indicator").classList.toggle("hidden", state.user.role !== "admin");
  renderMode();
}

function startEntrance() {
  showIntro();
}

function skipIntro() {
  showAuth();
}

function modeLabel(mode = state.mode) {
  return mode === "general" ? "일반 법령" : "시나리오";
}

function renderMode() {
  const isGeneral = state.mode === "general";
  $("mode-scenario").className = ["mode-button", isGeneral ? "" : "mode-button-active"].join(" ");
  $("mode-general").className = ["mode-button", isGeneral ? "mode-button-active" : ""].join(" ");
  $("mode-indicator").textContent = isGeneral ? "일반 법령" : "시나리오 상담";
  $("scenario-open").disabled = isGeneral;
  $("scenario-open").classList.toggle("is-disabled", isGeneral);
  $("question").placeholder = isGeneral ? "법령 질문을 입력하세요" : "사고 시나리오를 바탕으로 질문하세요";
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

function sourceLine(source, index, isAdmin) {
  const score = isAdmin && source.score !== undefined ? ` score=${source.score}` : "";
  return `${index + 1}. [${source.source_type || "source"}] ${source.law_name || ""} ${source.article || ""} ${source.page || ""}${score}`;
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

function appendMessage(message) {
  $("empty-state")?.classList.add("hidden");

  const row = document.createElement("article");
  if (message.id) row.dataset.messageId = String(message.id);
  row.className = [
    "message-row",
    message.role === "user" ? "message-row-user" : "message-row-assistant",
    message.status ? "message-row-status" : "",
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
  content.textContent = message.content;
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
    body.appendChild(actions);
  }

  if (message.role === "assistant" && message.payload) {
    if (state.user.role === "admin" && message.payload.cli_output) {
      const pre = document.createElement("pre");
      pre.className = "cli-output";
      pre.textContent = message.payload.cli_output;
      body.appendChild(pre);
    }
  }

  row.appendChild(avatar);
  row.appendChild(body);
  $("messages").appendChild(row);
  $("messages").scrollTop = $("messages").scrollHeight;
  return row;
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    if (window.Swal) {
      await Swal.fire({
        icon: "success",
        title: "복사 완료",
        text: "클립보드에 복사했습니다.",
        timer: 900,
        showConfirmButton: false,
        background: "#ffffff",
        color: "#1d2935",
      });
    }
  } catch {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  }
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
      else await createConversation();
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

async function createConversation() {
  const title = state.mode === "general" ? "새 일반 법령 상담" : "새 시나리오 상담";
  const data = await api("/api/conversations", {
    method: "POST",
    body: JSON.stringify({ title, mode: state.mode }),
  });
  state.conversationId = data.conversation.id;
  state.mode = data.conversation.mode || state.mode;
  await refreshConversations();
  renderEmptyState();
  $("chat-title").textContent = data.conversation.title;
  renderMode();
  setEmptyState();
}

async function loadConversation(id) {
  const data = await api(`/api/conversations/${id}`);
  state.conversationId = data.conversation.id;
  state.mode = data.conversation.mode || "scenario";
  $("chat-title").textContent = data.conversation.title;
  $("messages").innerHTML = "";
  data.messages.forEach(appendMessage);
  if (!data.messages.length) {
    renderEmptyState();
  }
  renderConversations();
  renderMode();
  setEmptyState();
}

function renderEmptyState() {
  $("messages").innerHTML = emptyStateHtml(state.mode);
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
}

async function bootstrap() {
  const data = await api("/api/me");
  state.user = data.user;
  if (!state.user) {
    showAuth();
    return;
  }
  showApp();
  await refreshConversations();
  const preferred = state.conversations.find((conv) => (conv.mode || "scenario") === state.mode);
  if (preferred) await loadConversation(preferred.id);
  else if (state.conversations.length && window.location.pathname !== "/general") await loadConversation(state.conversations[0].id);
  else await createConversation();
  await loadScenario();
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
    await bootstrap();
  } catch (error) {
    showAuth(error.message);
  }
});

$("intro-skip").addEventListener("click", skipIntro);

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
    $("auth-message").textContent = "일반 계정이 생성되었습니다. 로그인하세요.";
  } catch (error) {
    $("auth-message").textContent = error.message;
  }
});

$("logout").addEventListener("click", async () => {
  await api("/api/logout", { method: "POST", body: "{}" });
  state.user = null;
  state.conversations = [];
  state.conversationId = null;
  showAuth();
});

$("new-chat").addEventListener("click", async () => {
  closeSidebarOnMobile();
  await createConversation();
});

document.addEventListener("click", (event) => {
  if (!event.target.closest(".conversation-menu") && !event.target.closest(".conversation-menu-button")) {
    closeConversationMenus();
  }
});

$("mode-scenario").addEventListener("click", async () => {
  if (state.mode === "scenario") return;
  state.mode = "scenario";
  window.history.replaceState({}, "", "/");
  renderMode();
  await createConversation();
});

$("mode-general").addEventListener("click", async () => {
  if (state.mode === "general") return;
  state.mode = "general";
  window.history.replaceState({}, "", "/general");
  $("scenario-panel").classList.add("hidden");
  renderMode();
  await createConversation();
});

$("sidebar-toggle").addEventListener("click", () => {
  $("sidebar").classList.toggle("sidebar-open");
});

function closeSidebarOnMobile() {
  $("sidebar").classList.remove("sidebar-open");
}

$("scenario-open").addEventListener("click", async () => {
  if (state.mode === "general") return;
  await loadScenario();
  $("scenario-panel").classList.remove("hidden");
});

$("scenario-close").addEventListener("click", () => {
  $("scenario-panel").classList.add("hidden");
});

$("scenario-save").addEventListener("click", async () => {
  try {
    await api("/api/scenario", {
      method: "POST",
      body: JSON.stringify({
        overview: $("scenario-overview").value,
        details: $("scenario-details").value,
        workers: $("scenario-workers").value,
      }),
    });
    $("scenario-message").textContent = "저장했습니다.";
  } catch (error) {
    $("scenario-message").textContent = error.message;
  }
});

$("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;
  $("question").value = "";
  $("question").style.height = "auto";

  const pendingMessage = appendMessage({ role: "user", content: question, created_at: new Date().toISOString() });
  let statusMessage = appendMessage({ role: "assistant", content: "생각중 . . .", created_at: new Date().toISOString(), status: true });
  $("send").disabled = true;

  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ conversation_id: state.conversationId, question, mode: state.mode }),
    });
    state.conversationId = data.conversation_id;
    state.mode = data.mode || state.mode;
    pendingMessage.remove();
    statusMessage.remove();
    if (data.user_message) appendMessage(data.user_message);
    statusMessage = appendMessage({ role: "assistant", content: "정리 완료!", created_at: new Date().toISOString(), status: true });
    await wait(650);
    statusMessage.remove();
    appendMessage(data.message);
    await refreshConversations();
    renderMode();
  } catch (error) {
    statusMessage.remove();
    const message = error.message || "모델 서버와 연결하지 못했습니다.";
    if (window.Swal) {
      await Swal.fire({
        icon: "error",
        title: "모델 연결 실패",
        text: "Colab LLM 서버 URL(.env의 LLM_API_BASE) 또는 모델 서버 상태를 확인하세요.",
        confirmButtonText: "확인",
        confirmButtonColor: "#233f73",
        background: "#ffffff",
        color: "#1d2935",
        footer: escapeHtml(message),
      });
    }
    appendMessage({ role: "assistant", content: `모델 연결 실패: 설정과 서버 상태를 확인하세요.\n${message}` });
  } finally {
    $("send").disabled = false;
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

startEntrance();
