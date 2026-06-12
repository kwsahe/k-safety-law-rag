const state = {
  user: null,
  conversations: [],
  conversationId: null,
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
}

function renderConversations() {
  const list = $("conversation-list");
  list.innerHTML = "";
  state.conversations.forEach((conv) => {
    const button = document.createElement("button");
    button.className = [
      "min-h-11 rounded-2xl px-4 py-3 text-left text-sm font-black transition",
      conv.id === state.conversationId
        ? "bg-lavender text-indigo-950 shadow-sm"
        : "bg-white/60 text-slate-500 hover:bg-white",
    ].join(" ");
    button.textContent = conv.title;
    button.onclick = () => loadConversation(conv.id);
    list.appendChild(button);
  });
}

function sourceLine(source, index, isAdmin) {
  const score = isAdmin && source.score !== undefined ? ` score=${source.score}` : "";
  return `${index + 1}. [${source.source_type || "source"}] ${source.law_name || ""} ${source.article || ""} ${source.page || ""}${score}`;
}

function appendMessage(message) {
  const wrap = document.createElement("article");
  wrap.className = [
    "max-w-[min(900px,92%)] rounded-[26px] border px-5 py-4 leading-7 shadow-sm whitespace-pre-wrap",
    message.role === "user"
      ? "justify-self-end border-peach/70 bg-peach/70 text-rose-950"
      : "justify-self-start border-white/90 bg-white/85 text-ink",
  ].join(" ");
  wrap.textContent = message.content;
  if (message.role === "assistant" && message.payload) {
    if (state.user.role === "admin" && message.payload.cli_output) {
      const pre = document.createElement("pre");
      pre.className = "cli-output mt-4 overflow-auto rounded-2xl border border-slate-200 bg-slate-900 p-4 text-sm leading-6 text-slate-100 whitespace-pre-wrap";
      pre.textContent = message.payload.cli_output;
      wrap.appendChild(pre);
    } else if (message.payload.sources?.length) {
      const sources = document.createElement("div");
      sources.className = "mt-4 grid gap-2 border-t border-slate-100 pt-4 text-sm font-semibold text-slate-500 whitespace-normal";
      sources.innerHTML = message.payload.sources.map((source, index) => `<div class="rounded-2xl bg-skysoft/50 px-3 py-2">${escapeHtml(sourceLine(source, index, false))}</div>`).join("");
      wrap.appendChild(sources);
    }
  }
  $("messages").appendChild(wrap);
  $("messages").scrollTop = $("messages").scrollHeight;
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
    body: JSON.stringify({ title: "새 상담" }),
  });
  state.conversationId = data.conversation.id;
  await refreshConversations();
  $("messages").innerHTML = "";
  $("chat-title").textContent = data.conversation.title;
}

async function loadConversation(id) {
  const data = await api(`/api/conversations/${id}`);
  state.conversationId = data.conversation.id;
  $("chat-title").textContent = data.conversation.title;
  $("messages").innerHTML = "";
  data.messages.forEach(appendMessage);
  renderConversations();
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
  if (state.conversations.length) {
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

$("scenario-open").addEventListener("click", async () => {
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
  appendMessage({ role: "user", content: question });
  $("send").disabled = true;
  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ conversation_id: state.conversationId, question }),
    });
    state.conversationId = data.conversation_id;
    appendMessage(data.message);
    await refreshConversations();
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
