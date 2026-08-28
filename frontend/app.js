(() => {
  "use strict";

  const API = "/api";
  const state = {
    // sessionStorage, not localStorage: login is per-tab, not shared across
    // every tab in the browser - a brand-new tab starts logged out instead
    // of silently inheriting whichever user was last logged in elsewhere,
    // and two tabs can be logged in as two different users at once.
    userId: sessionStorage.getItem("docqa_user_id") || null,
    sessions: [],
    activeSessionId: null,
  };

  const el = (id) => document.getElementById(id);

  async function api(path, options = {}) {
    const headers = Object.assign({}, options.headers || {});
    if (state.userId) headers["X-User-Id"] = state.userId;
    if (options.body && !(options.body instanceof FormData)) {
      headers["Content-Type"] = "application/json";
    }
    const response = await fetch(`${API}${path}`, { ...options, headers });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        detail = body.detail || detail;
      } catch (_) {
        /* ignore */
      }
      throw new Error(detail);
    }
    if (response.status === 204) return null;
    return response.json();
  }

  // --- Login ---
  function showLogin() {
    el("login-screen").classList.remove("hidden");
    el("app-screen").classList.add("hidden");
  }

  function showApp() {
    el("login-screen").classList.add("hidden");
    el("app-screen").classList.remove("hidden");
    el("current-user").textContent = state.userId;
    loadSessions();
  }

  el("login-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const userId = el("login-user-id").value.trim();
    el("login-error").textContent = "";
    if (!userId) return;
    try {
      const res = await fetch(`${API}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId }),
      });
      if (!res.ok) throw new Error("Login failed");
      const data = await res.json();
      state.userId = data.user_id;
      sessionStorage.setItem("docqa_user_id", state.userId);
      showApp();
    } catch (err) {
      el("login-error").textContent = err.message;
    }
  });

  el("logout-btn").addEventListener("click", () => {
    state.userId = null;
    state.activeSessionId = null;
    sessionStorage.removeItem("docqa_user_id");
    showLogin();
  });

  // --- Sessions ---
  async function loadSessions() {
    const data = await api("/sessions");
    state.sessions = data.sessions;
    renderSessionList();
    if (state.sessions.length && !state.activeSessionId) {
      selectSession(state.sessions[0].session_id);
    }
  }

  function renderSessionList() {
    const list = el("session-list");
    list.innerHTML = "";
    for (const s of state.sessions) {
      const item = document.createElement("div");
      item.className = "session-item" + (s.session_id === state.activeSessionId ? " active" : "");
      item.textContent = s.title || "New chat";
      item.addEventListener("click", () => selectSession(s.session_id));
      list.appendChild(item);
    }
  }

  el("new-chat-btn").addEventListener("click", async () => {
    const session = await api("/sessions", {
      method: "POST",
      body: JSON.stringify({ title: "New chat" }),
    });
    state.sessions.unshift(session);
    renderSessionList();
    selectSession(session.session_id);
  });

  el("delete-session-btn").addEventListener("click", async () => {
    if (!state.activeSessionId) return;
    if (!confirm("Delete this chat and everything in it? This cannot be undone.")) return;
    await api(`/sessions/${state.activeSessionId}`, { method: "DELETE" });
    state.sessions = state.sessions.filter((s) => s.session_id !== state.activeSessionId);
    state.activeSessionId = null;
    renderSessionList();
    el("chat-area").classList.add("hidden");
    el("empty-state").classList.remove("hidden");
    if (state.sessions.length) selectSession(state.sessions[0].session_id);
  });

  async function selectSession(sessionId) {
    state.activeSessionId = sessionId;
    renderSessionList();
    el("empty-state").classList.add("hidden");
    el("chat-area").classList.remove("hidden");
    const session = state.sessions.find((s) => s.session_id === sessionId);
    el("session-title").textContent = (session && session.title) || "Chat";
    await Promise.all([loadMessages(), loadFiles()]);
  }

  // --- Messages ---
  async function loadMessages() {
    const data = await api(`/sessions/${state.activeSessionId}/messages`);
    const container = el("messages");
    container.innerHTML = "";
    for (const m of data.messages) renderMessage(m);
    container.scrollTop = container.scrollHeight;
  }

  function renderMessage(m) {
    const container = el("messages");
    const div = document.createElement("div");
    div.className = "message " + (m.role === "user" ? "user" : "assistant");
    div.textContent = m.content;

    if (m.role === "assistant" && m.question_type) {
      const meta = document.createElement("div");
      meta.className = "message-meta";
      meta.textContent = m.question_type;
      div.appendChild(meta);
    }
    if (m.sources && m.sources.length) {
      const src = document.createElement("div");
      src.className = "message-sources";
      for (const s of m.sources) {
        const line = document.createElement("div");
        line.textContent = formatSource(s);
        if (line.textContent) src.appendChild(line);
      }
      div.appendChild(src);
    }
    container.appendChild(div);
  }

  function formatSource(s) {
    if (s.type === "structured_query") {
      return `Source: database query (${s.row_count} row${s.row_count === 1 ? "" : "s"})`;
    }
    if (s.filename) {
      const bits = [];
      if (s.page_start) bits.push(`page ${s.page_start}`);
      if (s.slide_number) bits.push(`slide ${s.slide_number}`);
      if (s.section) bits.push(`section: ${s.section}`);
      return `Source: ${s.filename}${bits.length ? " (" + bits.join(", ") + ")" : ""}`;
    }
    return "";
  }

  el("chat-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = el("chat-input");
    const question = input.value.trim();
    if (!question || !state.activeSessionId) return;
    input.value = "";
    renderMessage({ role: "user", content: question });
    el("messages").scrollTop = el("messages").scrollHeight;

    try {
      const response = await api(`/sessions/${state.activeSessionId}/chat`, {
        method: "POST",
        body: JSON.stringify({ question }),
      });
      renderMessage({
        role: "assistant",
        content: response.answer,
        question_type: response.question_type,
        sources: response.sources,
      });
    } catch (err) {
      renderMessage({ role: "assistant", content: `Error: ${err.message}` });
    }
    el("messages").scrollTop = el("messages").scrollHeight;
  });

  // --- Files ---
  async function loadFiles() {
    const data = await api(`/sessions/${state.activeSessionId}/files`);
    renderFileList(data.files);
  }

  function renderFileList(files) {
    const container = el("file-list");
    container.innerHTML = "";
    for (const f of files) {
      const chip = document.createElement("span");
      chip.className = "file-chip " + f.status;
      chip.textContent = `${f.filename} (${f.status})`;
      container.appendChild(chip);
    }
  }

  el("file-input").addEventListener("change", async (e) => {
    const files = Array.from(e.target.files || []);
    e.target.value = "";
    for (const file of files) {
      const formData = new FormData();
      formData.append("file", file);
      try {
        await api(`/sessions/${state.activeSessionId}/files`, { method: "POST", body: formData });
      } catch (err) {
        alert(`Failed to upload ${file.name}: ${err.message}`);
      }
      await loadFiles();
    }
  });

  // --- Init ---
  if (state.userId) {
    showApp();
  } else {
    showLogin();
  }
})();
