const state = {
  user: null,
  conversations: [],
  conversationId: null,
  mode: window.location.pathname === "/general" ? "general" : "scenario",
};

const $ = (id) => document.getElementById(id);

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

function showAuth(message = "") {
  $("auth").classList.remove("hidden");
  $("app").classList.add("hidden");
  $("auth-message").textContent = message;
}

function showApp() {
  $("auth").classList.add("hidden");
  $("app").classList.remove("hidden");
  $("user-badge").textContent = `${state.user.username} · ${state.user.role}`;
  $("admin-indicator").classList.toggle("hidden", state.user.role !== "admin");
  renderMode();
}

function modeLabel(mode = state.mode) {
  return mode === "general" ? "일반 법령" : "시나리오";
}

function renderMode() {
  const isGeneral = state.mode === "general";
  $("mode-scenario").className = [
    "min-h-10 rounded-full px-3 text-sm font-black transition",
    isGeneral ? "text-mutedBlue hover:bg-white/70" : "bg-white/90 text-oceanDeep shadow-sm",
  ].join(" ");
  $("mode-general").className = [
    "min-h-10 rounded-full px-3 text-sm font-black transition",
    isGeneral ? "bg-white/90 text-oceanDeep shadow-sm" : "text-mutedBlue hover:bg-white/70",
  ].join(" ");
  $("mode-indicator").textContent = isGeneral ? "일반 법령 상담" : "시나리오 상담";
  $("mode-indicator").className = [
    "rounded-full px-4 py-2 text-xs font-black",
    isGeneral ? "bg-[#edf2f5] text-slate-600" : "bg-oceanSoft text-oceanDeep",
  ].join(" ");
  $("scenario-open").disabled = isGeneral;
  $("scenario-open").classList.toggle("opacity-50", isGeneral);
  $("scenario-open").classList.toggle("cursor-not-allowed", isGeneral);
  $("question").placeholder = isGeneral
    ? "시나리오 없이 법령 질문을 입력하세요"
    : "사고 시나리오를 바탕으로 질문하세요";
}

function renderConversations() {
  const list = $("conversation-list");
  list.innerHTML = "";
  state.conversations.forEach((conv) => {
    const item = document.createElement("div");
    item.className = [
      "relative rounded-[26px] px-4 py-4 transition",
      conv.id === state.conversationId
        ? "border border-blueLine/75 bg-white/90 text-graphite shadow-soft"
        : "border border-blueLine/45 bg-oceanSoft/38 text-mutedBlue hover:bg-white/85 hover:shadow-sm",
    ].join(" ");
    item.innerHTML = `
      <div class="grid grid-cols-[1fr_auto] items-start gap-2">
        <button class="conversation-open min-w-0 text-left text-sm font-black" type="button">
          <span class="block truncate">${escapeHtml(conv.title)}</span>
          <span class="mt-1 inline-block rounded-full border border-blueLine/45 bg-white/70 px-2 py-0.5 text-[11px] font-black text-mutedBlue">${modeLabel(conv.mode)}</span>
        </button>
        <button class="conversation-menu-button min-h-8 min-w-8 rounded-full border border-blueLine/50 bg-oceanSoft/70 text-sm font-black text-mutedBlue hover:bg-white hover:text-oceanDeep" type="button" aria-label="채팅 메뉴">...</button>
      </div>
      <div class="conversation-menu hidden absolute right-4 top-12 z-20 min-w-36 rounded-[24px] border border-blueLine/75 bg-glass p-2 text-sm font-black text-slate-600 shadow-float">
        <button class="conversation-rename flex min-h-9 w-full items-center rounded-xl px-3 text-left hover:bg-slate-50" type="button">이름 수정</button>
        <button class="conversation-delete flex min-h-9 w-full items-center rounded-xl px-3 text-left text-danger hover:bg-rose-50" type="button">채팅 삭제</button>
      </div>
    `;
    item.querySelector(".conversation-open").onclick = () => loadConversation(conv.id);
    item.querySelector(".conversation-rename").onclick = (event) => {
      event.stopPropagation();
      closeConversationMenus();
      renameConversation(conv);
    };
    item.querySelector(".conversation-menu-button").onclick = (event) => {
      event.stopPropagation();
      toggleConversationMenu(item);
    };
    item.querySelector(".conversation-delete").onclick = (event) => {
      event.stopPropagation();
      closeConversationMenus();
      deleteConversation(conv);
    };
    list.appendChild(item);
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

function appendMessage(message) {
  const wrap = document.createElement("article");
  if (message.id) wrap.dataset.messageId = String(message.id);
  wrap.className = [
    "message-bubble max-w-[min(900px,92%)] rounded-[28px] border px-5 py-4 leading-7 shadow-sm whitespace-pre-wrap",
    message.role === "user" ? "message-bubble-user justify-self-end" : "message-bubble-assistant justify-self-start",
  ].join(" ");
  const time = document.createElement("div");
  time.className = [
    "message-time mb-2 text-[11px] font-black tracking-normal",
  ].join(" ");
  time.textContent = `${message.role === "user" ? "입력 시간" : "출력 시간"} ${formatMessageTime(message.created_at)}`;
  wrap.appendChild(time);

  const content = document.createElement("div");
  content.className = "message-content";
  content.textContent = message.content;
  wrap.appendChild(content);

  const actions = document.createElement("div");
  actions.className = "mt-3 flex justify-end gap-2";
  const copyButton = document.createElement("button");
  copyButton.className = [
    "copy-button min-h-8 rounded-full px-3 text-xs font-black transition",
  ].join(" ");
  copyButton.type = "button";
  copyButton.textContent = "복사";
  copyButton.onclick = () => copyText(message.content);
  actions.appendChild(copyButton);
  wrap.appendChild(actions);

  if (message.role === "assistant" && message.payload) {
    if (state.user.role === "admin" && message.payload.cli_output) {
      const pre = document.createElement("pre");
      pre.className = "cli-output mt-4 overflow-auto rounded-2xl border border-blueLine/60 bg-[#172331] p-4 text-sm leading-6 text-slate-100 whitespace-pre-wrap";
      pre.textContent = message.payload.cli_output;
      wrap.appendChild(pre);
    } else if (message.payload.sources?.length) {
      const sources = document.createElement("div");
      sources.className = "mt-4 grid gap-2 border-t border-blueLine/45 pt-4 text-sm font-semibold text-mutedBlue whitespace-normal";
      sources.innerHTML = message.payload.sources.map((source, index) => `<div class="rounded-2xl border border-blueLine/45 bg-oceanSoft/60 px-3 py-2">${escapeHtml(sourceLine(source, index, false))}</div>`).join("");
      wrap.appendChild(sources);
    }
  }
  $("messages").appendChild(wrap);
  $("messages").scrollTop = $("messages").scrollHeight;
  return wrap;
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
        color: "#293241",
      });
    }
  } catch (error) {
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
        confirmButtonColor: "#2dd4bf",
        background: "#ffffff",
        color: "#293241",
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
        text: `"${conv.title}" 상담을 화면에서 삭제합니다. DB에는 삭제 로그가 보존됩니다.`,
        showCancelButton: true,
        confirmButtonText: "삭제",
        cancelButtonText: "취소",
        confirmButtonColor: "#fb7185",
        background: "#ffffff",
        color: "#293241",
      })
    : { isConfirmed: confirm(`"${conv.title}" 채팅을 삭제할까요? DB에는 삭제 로그가 보존됩니다.`) };
  if (!result.isConfirmed) return;
  try {
    await api(`/api/conversations/${conv.id}`, { method: "DELETE" });
    const wasActive = state.conversationId === conv.id;
    await refreshConversations();
    if (wasActive) {
      const next = state.conversations.find((item) => (item.mode || "scenario") === state.mode) || state.conversations[0];
      if (next) {
        await loadConversation(next.id);
      } else {
        await createConversation();
      }
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
      confirmButtonColor: "#2dd4bf",
      background: "#ffffff",
      color: "#293241",
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
  const data = await api("/api/conversations", {
    method: "POST",
    body: JSON.stringify({ title: "새 상담", mode: state.mode }),
  });
  state.conversationId = data.conversation.id;
  state.mode = data.conversation.mode || state.mode;
  await refreshConversations();
  $("messages").innerHTML = "";
  $("chat-title").textContent = data.conversation.title;
  renderMode();
}

async function loadConversation(id) {
  const data = await api(`/api/conversations/${id}`);
  state.conversationId = data.conversation.id;
  state.mode = data.conversation.mode || "scenario";
  $("chat-title").textContent = data.conversation.title;
  $("messages").innerHTML = "";
  data.messages.forEach(appendMessage);
  renderConversations();
  renderMode();
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
  if (preferred) {
    await loadConversation(preferred.id);
  } else if (state.conversations.length && window.location.pathname !== "/general") {
    await loadConversation(state.conversations[0].id);
  } else {
    await createConversation();
  }
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
    $("auth-message").textContent = "일반 계정을 생성했습니다. 로그인하세요.";
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

$("new-chat").addEventListener("click", createConversation);

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
  const pendingMessage = appendMessage({ role: "user", content: question, created_at: new Date().toISOString() });
  $("send").disabled = true;
  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ conversation_id: state.conversationId, question, mode: state.mode }),
    });
    state.conversationId = data.conversation_id;
    state.mode = data.mode || state.mode;
    pendingMessage.remove();
    if (data.user_message) appendMessage(data.user_message);
    appendMessage(data.message);
    await refreshConversations();
    renderMode();
  } catch (error) {
    const message = error.message || "모델 서버와 연결하지 못했습니다.";
    if (window.Swal) {
      await Swal.fire({
        icon: "error",
        title: "모델 연결 실패",
        text: "Colab LLM 서버 URL(.env의 LLM_API_BASE) 또는 모델 서버 상태를 확인하세요.",
        confirmButtonText: "확인",
        confirmButtonColor: "#2dd4bf",
        background: "#ffffff",
        color: "#293241",
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
  event.target.style.height = `${Math.min(event.target.scrollHeight, 160)}px`;
});

bootstrap().catch((error) => showAuth(error.message));
