const state = { models: [], model_files: [], model_functions: [], connections: [], users: [], history_logs: [], streams: [], memory: null };
const streamDrafts = {};
let streamControlFocused = false;
let refreshTimer = null;

const shell = document.querySelector(".shell");
const savedSidebar = localStorage.getItem("sidebarCollapsed");
if (savedSidebar === "1") shell.classList.add("sidebar-collapsed");
const content = document.querySelector(".content");
const canOperate = ["admin", "operator"].includes(window.CURRENT_ROLE);
const canAdmin = window.CURRENT_ROLE === "admin";

const api = async (url, options = {}) => {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    if (response.status === 401) {
      stopRefreshLoop();
      if (window.location.pathname !== "/login") window.location.replace("/login");
      throw new Error("登录已失效，请重新登录");
    }
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    const message = response.status === 405 ? "接口方法不支持，请重启后端服务后再试" : formatApiError(detail.detail || response.statusText);
    throw new Error(message);
  }
  if (response.status === 204) return null;
  return response.json();
};

const postClientLog = (event, payload = {}) => {
  fetch("/api/client-log", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ event, ...payload }),
  }).catch(() => {});
};

const apiWithTimeout = async (url, options = {}, timeoutMs = 6000) => {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await api(url, { ...options, signal: controller.signal });
  } catch (err) {
    if (err.name === "AbortError") throw new Error(`请求超时（${timeoutMs}ms）`);
    throw err;
  } finally {
    clearTimeout(timer);
  }
};

const upload = async (url, field, file) => {
  const formData = new FormData();
  formData.append(field, file);
  const response = await fetch(url, { method: "POST", body: formData });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(formatApiError(detail.detail || response.statusText));
  }
  return response.json();
};

function formatApiError(detail) {
  if (Array.isArray(detail)) {
    return detail.map((item) => `${(item.loc || []).join(".")}: ${item.msg}`).join("\n");
  }
  if (typeof detail === "object" && detail !== null) return JSON.stringify(detail);
  return String(detail);
}

function formatMemory(memory) {
  if (!memory || !memory.total_mb) return "-";
  const usedGb = memory.used_mb / 1024;
  const totalGb = memory.total_mb / 1024;
  return `${usedGb.toFixed(1)}/${totalGb.toFixed(1)} GB`;
}

const modelOptions = (selected, includeNone = false) => `${includeNone ? `<option value="" ${!selected ? "selected" : ""}>无模型</option>` : ""}${state.models.map((m) => `<option value="${m.id}" ${m.id === selected ? "selected" : ""}>${m.name}</option>`).join("")}`;
const connectionOptions = (selected) => state.connections.map((c) => `<option value="${c.id}" ${c.id === selected ? "selected" : ""}>${c.name}</option>`).join("");
const streamOptions = (selected) => state.streams.map((s) => `<option value="${s.id}" ${Number(selected) === s.id ? "selected" : ""}>${s.id}</option>`).join("");
const byId = (id) => document.getElementById(id);
const findStream = (id) => state.streams.find((s) => String(s.id) === String(id));
const streamControlIsLocked = (stream) => stream?.running || !canOperate;
const hasRunningStreams = () => state.streams.some((stream) => stream.running);

async function refresh(options = {}) {
  const snapshot = await api("/api/snapshot");
  Object.assign(state, snapshot);
  syncRefreshLoop();
  if (!options.forceRender && (streamControlFocused || document.querySelector("dialog[open]"))) return;
  if (options.statsOnly) {
    renderMetrics();
    updateStreamStats();
    return;
  }
  render();
}

function currentView() {
  return document.querySelector(".view.active")?.id || "monitor";
}

function startRefreshLoop() {
  if (refreshTimer) return;
  refreshTimer = setInterval(() => refresh({ statsOnly: true }), 3000);
}

function stopRefreshLoop() {
  if (!refreshTimer) return;
  clearInterval(refreshTimer);
  refreshTimer = null;
}

function syncRefreshLoop() {
  if (currentView() === "monitor" && hasRunningStreams()) {
    startRefreshLoop();
  } else {
    stopRefreshLoop();
  }
}

function scheduleResourceRefresh() {
  [0, 800, 2000, 5000].forEach((delay) => setTimeout(() => refresh({ forceRender: true }), delay));
}

function render() {
  renderMetrics();
  renderStreams();
  renderConnections();
  renderModelFiles();
  renderModelFunctions();
  renderUsers();
  renderHistoryLogs();
  fillModelSelects();
}

function renderMetrics() {
  byId("metric-online").textContent = `${state.streams.filter((s) => s.running).length}/${state.streams.length}`;
  byId("metric-models").textContent = state.models.length;
  byId("metric-connections").textContent = state.connections.length;
  byId("metric-memory").textContent = formatMemory(state.memory);
  byId("metric-memory-label").textContent = `${state.memory?.label || "显存"}占用`;
  byId("metric-role").textContent = roleName(window.CURRENT_ROLE);
}

function applyViewTheme(view) {
  content.classList.remove("theme-monitor", "theme-connections", "theme-models", "theme-users", "theme-logs", "theme-settings");
  content.classList.add(`theme-${view}`);
  shell.classList.remove("theme-monitor", "theme-connections", "theme-models", "theme-users", "theme-logs", "theme-settings");
  shell.classList.add(`theme-${view}`);
}

function renderStreams() {
  const grid = byId("stream-grid");
  if (!grid) return;
  grid.innerHTML = state.streams.map((stream) => {
    const running = stream.running ? "running" : "";
    const fpsOverlay = stream.running ? `<span class="video-fps">FPS: ${Number(stream.fps || 0).toFixed(1)}</span>` : "";
    const draft = streamDrafts[stream.id] || {};
    const selectedConnection = stream.connection_id || draft.connection_id || state.connections[0]?.id || "";
    const selectedModel = stream.model_id ?? draft.model_id ?? state.connections.find((c) => c.id === selectedConnection)?.default_model_id ?? "";
    const configDisabled = streamControlIsLocked(stream) ? "disabled" : "";
    const startDisabled = canOperate && !stream.running ? "" : "disabled";
    const stopDisabled = canOperate && stream.running ? "" : "disabled";
    streamDrafts[stream.id] = { connection_id: selectedConnection, model_id: selectedModel };
    return `
      <article class="stream-card" data-stream="${stream.id}">
        <div class="stream-head"><h3>通道 ${stream.id}</h3><span class="status stream-status ${running}">${stream.running ? "在线" : "离线"}</span></div>
        <div class="video-box">${stream.running ? `<img data-stream="${stream.id}" src="/api/video/${stream.id}" alt="通道 ${stream.id}">${fpsOverlay}` : "未启动"}</div>
        <div class="stream-controls">
          <select class="stream-connection" data-stream="${stream.id}" ${configDisabled}>${connectionOptions(selectedConnection)}</select>
          <select class="stream-model" data-stream="${stream.id}" ${configDisabled}>${modelOptions(selectedModel, true)}</select>
          <button data-action="start" data-stream="${stream.id}" ${startDisabled}>启动</button>
          <button class="ghost" data-action="stop" data-stream="${stream.id}" ${stopDisabled}>停止</button>
          <button class="danger" data-action="delete-stream" data-stream="${stream.id}" ${canAdmin && !stream.running ? "" : "disabled"}>删除</button>
        </div>
        <div class="stream-meta">
          FPS: ${stream.fps} · 帧数: ${stream.frames}<br>
          RTSP: ${stream.rtsp_url}<br>
          ${stream.last_error ? `<span class="danger-text">错误: ${stream.last_error}</span>` : ""}
        </div>
      </article>
    `;
  }).join("");
  logMonitorImageMetrics();
}

function updateStreamStats() {
  let needsRender = false;
  state.streams.forEach((stream) => {
    const card = document.querySelector(`.stream-card[data-stream="${stream.id}"]`);
    if (!card) return;
    const hasVideo = Boolean(card.querySelector(".video-box img"));
    if (hasVideo !== Boolean(stream.running)) {
      needsRender = true;
      return;
    }
    card.querySelector(".stream-status").textContent = stream.running ? "在线" : "离线";
    card.querySelector(".stream-status").classList.toggle("running", Boolean(stream.running));
    const fps = card.querySelector(".video-fps");
    if (fps) fps.textContent = `FPS: ${Number(stream.fps || 0).toFixed(1)}`;
    const meta = card.querySelector(".stream-meta");
    if (meta) {
      meta.innerHTML = `
          FPS: ${stream.fps} · 帧数: ${stream.frames}<br>
          RTSP: ${stream.rtsp_url}<br>
          ${stream.last_error ? `<span class="danger-text">错误: ${stream.last_error}</span>` : ""}
        `;
    }
  });
  if (needsRender) render();
}

function logMonitorImageMetrics() {
  document.querySelectorAll("#monitor .video-box img").forEach((img) => {
    if (img.dataset.metricsBound === "1") return;
    img.dataset.metricsBound = "1";
    img.addEventListener("load", () => {
      const box = img.closest(".video-box")?.getBoundingClientRect();
      const payload = {
        stream: img.dataset.stream,
        natural: `${img.naturalWidth}x${img.naturalHeight}`,
        rendered: `${Math.round(img.clientWidth)}x${Math.round(img.clientHeight)}`,
        box: box ? `${Math.round(box.width)}x${Math.round(box.height)}` : "",
        objectFit: getComputedStyle(img).objectFit,
      };
      console.debug("monitor video layout", payload);
      postClientLog("monitor_video_layout", payload);
    });
    img.addEventListener("error", () => {
      postClientLog("monitor_video_error", { stream: img.dataset.stream, src: img.getAttribute("src") });
    });
  });
}

function renderConnections() {
  const rows = byId("connection-rows");
  if (!rows) return;
  rows.innerHTML = state.connections.map((c) => `
    <tr>
      <td>${c.status}</td><td>${c.name}</td><td>${c.source}</td><td>${c.type}</td>
      <td>${modelName(c.default_model_id)}</td><td>${c.default_stream_id}</td>
      <td class="actions">
        <button class="ghost" data-action="test-connection" data-id="${c.id}" ${canOperate ? "" : "disabled"}>测试</button>
        <button class="ghost" data-action="edit-connection" data-id="${c.id}" ${canOperate ? "" : "disabled"}>编辑</button>
        <button class="danger" data-action="delete-connection" data-id="${c.id}" ${canOperate ? "" : "disabled"}>删除</button>
      </td>
    </tr>
  `).join("");
}

function renderModelFiles() {
  const rows = byId("model-file-rows");
  if (!rows) return;
  rows.innerHTML = state.model_files.map((file) => `
    <tr>
      <td class="cell-entry" title="${escapeAttr(file.name)}">${file.name}</td>
      <td>${formatBytes(file.size)}</td>
      <td>${formatTime(file.modified_time)}</td>
      <td class="actions"><button class="ghost" data-action="view-model-file" data-name="${escapeAttr(file.name)}">查看</button></td>
    </tr>
  `).join("");
}

function renderModelFileDetail(detail) {
  const labels = detail.labels || [];
  byId("model-file-detail-title").textContent = `${detail.name} 详情`;
  byId("model-file-detail-body").innerHTML = `
    <div class="detail-grid">
      <div><strong>加载状态</strong><span>${detail.loaded ? "已加载" : "未加载"}</span></div>
      <div><strong>Warmup</strong><span>${detail.warmup_done ? "已完成" : "未完成"}</span></div>
      <div><strong>设备</strong><span>${detail.device || "-"}</span></div>
      <div><strong>文件大小</strong><span>${formatBytes(detail.size)}</span></div>
      <div><strong>显存占用</strong><span>${formatBytes((detail.memory_allocated_mb || 0) * 1024 * 1024)}</span></div>
      <div><strong>显存保留</strong><span>${formatBytes((detail.memory_reserved_mb || 0) * 1024 * 1024)}</span></div>
    </div>
    ${detail.error ? `<p class="danger-text">错误: ${detail.error}</p>` : ""}
    <table class="label-table">
      <thead><tr><th>ID</th><th>标注名称</th></tr></thead>
      <tbody>${labels.map((label) => `<tr><td>${label.id}</td><td>${label.name}</td></tr>`).join("") || `<tr><td colspan="2">无标注信息</td></tr>`}</tbody>
    </table>
  `;
  byId("model-file-detail-dialog").showModal();
}

function renderModelFunctions() {
  const rows = byId("model-function-rows");
  if (!rows) return;
  rows.innerHTML = state.model_functions.map((m) => `
    <tr>
      <td class="cell-id" title="${m.id}">${m.id}</td>
      <td class="cell-name" title="${m.name}">${m.name}</td>
      <td class="cell-task">${m.task}</td>
      <td class="cell-entry" title="${m.entrypoint}">${m.entrypoint}</td>
      <td class="cell-config"><code title="${escapeAttr(JSON.stringify(m.config))}">${shortConfig(m.config)}</code></td>
      <td class="cell-status">${m.enabled ? "启用" : "禁用"}</td>
      <td class="actions">
        <button class="ghost" data-action="edit-model-function" data-id="${m.id}" ${canAdmin ? "" : "disabled"}>编辑</button>
        <button class="ghost" data-action="upload-pt" data-id="${m.id}" ${canAdmin ? "" : "disabled"}>上传PT</button>
        <button class="ghost" data-action="upload-code" data-id="${m.id}" ${canAdmin ? "" : "disabled"}>上传代码</button>
        <button class="danger" data-action="delete-model-function" data-id="${m.id}" ${canAdmin ? "" : "disabled"}>删除</button>
        <input class="upload-input" data-kind="pt" data-id="${m.id}" type="file" accept=".pt" hidden>
        <input class="upload-input" data-kind="code" data-id="${m.id}" type="file" accept=".py" hidden>
      </td>
    </tr>
  `).join("");
}

function formatBytes(value) {
  const size = Number(value || 0);
  if (size >= 1024 * 1024 * 1024) return `${(size / 1024 / 1024 / 1024).toFixed(2)} GB`;
  if (size >= 1024 * 1024) return `${(size / 1024 / 1024).toFixed(2)} MB`;
  if (size >= 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${size} B`;
}

function shortConfig(config) {
  const text = JSON.stringify(config);
  return text.length > 120 ? `${text.slice(0, 120)}...` : text;
}

function escapeAttr(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function renderUsers() {
  const rows = byId("user-rows");
  if (!rows) return;
  rows.innerHTML = state.users.map((u) => `
    <tr>
      <td>${u.username}</td><td>${roleName(u.role)}</td><td>${u.enabled ? "启用" : "禁用"}</td><td>${u.last_login || "-"}</td>
      <td class="actions">
        <button class="ghost" data-action="edit-user" data-name="${u.username}" ${canAdmin ? "" : "disabled"}>编辑</button>
        <button class="danger" data-action="delete-user" data-name="${u.username}" ${canAdmin && u.username !== "admin" ? "" : "disabled"}>删除</button>
      </td>
    </tr>
  `).join("");
}

function renderHistoryLogs() {
  const rows = byId("history-log-rows");
  if (!rows) return;
  rows.innerHTML = state.history_logs.map((log) => `
    <tr>
      <td class="cell-time" title="${log.time}">${formatTime(log.time)}</td>
      <td>${log.user}</td>
      <td>${actionName(log.action)}</td>
      <td>${log.target_type}${log.target_id ? ` / ${log.target_id}` : ""}</td>
      <td><span class="result ${log.result}">${resultName(log.result)}</span></td>
      <td class="cell-message" title="${escapeAttr(log.message || "")}">${log.message || "-"}</td>
    </tr>
  `).join("");
}

function formatTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function actionName(action) {
  return {
    create: "新增",
    update: "编辑",
    delete: "删除",
    test: "测试",
    start: "启动",
    stop: "停止",
    switch_model: "切换模型",
    upload_pt: "上传PT",
    upload_code: "上传代码",
    login: "登录",
    add_stream: "新增通道",
    delete_stream: "删除通道",
  }[action] || action;
}

function resultName(result) {
  return result === "failed" ? "失败" : "成功";
}

function fillModelSelects() {
  document.querySelectorAll('select[name="default_model_id"]').forEach((select) => {
    const current = select.value;
    select.innerHTML = modelOptions(current);
  });
}

function modelName(id) {
  return state.models.find((m) => m.id === id)?.name || id;
}

function roleName(role) {
  return { admin: "管理员", operator: "操作员", viewer: "只读" }[role] || role;
}

function localizeCurrentUserBadge() {
  const badge = document.querySelector(".user-badge");
  if (!badge) return;
  const username = badge.dataset.username || badge.textContent.split("·")[0].trim();
  const role = badge.dataset.role || window.CURRENT_ROLE;
  badge.textContent = `${username} · ${roleName(role)}`;
}

document.querySelectorAll(".nav").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav,.view").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    byId(button.dataset.view).classList.add("active");
    byId("page-title").textContent = button.textContent;
    applyViewTheme(button.dataset.view);
    syncRefreshLoop();
    refresh();
  });
});

applyViewTheme(document.querySelector(".nav.active")?.dataset.view || "monitor");
localizeCurrentUserBadge();

byId("toggle-sidebar")?.addEventListener("click", () => {
  shell.classList.toggle("sidebar-collapsed");
  localStorage.setItem("sidebarCollapsed", shell.classList.contains("sidebar-collapsed") ? "1" : "0");
});

document.body.addEventListener("click", async (event) => {
  const clicked = event.target instanceof Element ? event.target : null;
  const target = clicked?.closest("[data-action], [data-dialog-close]");
  if (!target) return;
  const closeButton = target.closest("[data-dialog-close]");
  if (closeButton) {
    event.preventDefault();
    closeButton.closest("dialog")?.close();
    return;
  }
  const action = target.dataset.action;
  if (action) postClientLog("ui_action", { action, stream: target.dataset.stream || "", id: target.dataset.id || "", name: target.dataset.name || "" });
  try {
    if (action === "start") {
      const id = target.dataset.stream;
      const card = target.closest(".stream-card");
      const connectionId = card.querySelector(".stream-connection").value;
      const modelId = card.querySelector(".stream-model").value;
      document.activeElement?.blur();
      target.disabled = true;
      streamControlFocused = false;
      await api(`/api/streams/${id}/start`, { method: "POST", body: JSON.stringify({ connection_id: connectionId, model_id: modelId, rtsp_enabled: true }) });
      scheduleResourceRefresh();
    }
    if (action === "add-stream") {
      target.disabled = true;
      target.textContent = "新增中";
      try {
        const stream = await apiWithTimeout("/api/streams", { method: "POST", body: "{}" }, 6000);
        if (!state.streams.some((item) => item.id === stream.id)) state.streams.push(stream);
        render();
        return;
      } finally {
        target.disabled = !canAdmin;
        target.textContent = "新增通道";
      }
    }
    if (action === "stop") {
      document.activeElement?.blur();
      target.disabled = true;
      streamControlFocused = false;
      await api(`/api/streams/${target.dataset.stream}/stop`, { method: "POST", body: "{}" });
      scheduleResourceRefresh();
    }
    if (action === "delete-stream" && confirm(`删除通道 ${target.dataset.stream}？`)) {
      await api(`/api/streams/${target.dataset.stream}`, { method: "DELETE" });
    }
    if (action === "edit-connection") openConnectionDialog(state.connections.find((c) => c.id === target.dataset.id));
    if (action === "delete-connection" && confirm("删除该连接？")) await api(`/api/connections/${target.dataset.id}`, { method: "DELETE" });
    if (action === "test-connection") {
      target.disabled = true;
      target.textContent = "测试中";
      try {
        await apiWithTimeout(`/api/connections/${target.dataset.id}/test`, { method: "POST", body: "{}" }, 20000);
        alert("连接测试成功");
      } catch (err) {
        await refresh();
        throw err;
      } finally {
        target.disabled = false;
        target.textContent = "测试";
      }
    }
    if (action === "edit-user") openUserDialog(state.users.find((u) => u.username === target.dataset.name));
    if (action === "delete-user" && confirm("删除该用户？")) await api(`/api/users/${target.dataset.name}`, { method: "DELETE" });
    if (action === "edit-model-function") openModelFunctionDialog(state.model_functions.find((m) => m.id === target.dataset.id));
    if (action === "view-model-file") {
      const originalText = target.textContent;
      target.disabled = true;
      target.textContent = "加载中";
      try {
        const detail = await api(`/api/model-files/${encodeURIComponent(target.dataset.name)}/detail`);
        renderModelFileDetail(detail);
        return;
      } finally {
        target.disabled = false;
        target.textContent = originalText || "查看";
      }
    }
    if (action === "upload-pt") target.parentElement.querySelector(`input[data-kind="pt"][data-id="${target.dataset.id}"]`)?.click();
    if (action === "upload-code") target.parentElement.querySelector(`input[data-kind="code"][data-id="${target.dataset.id}"]`)?.click();
    if (action === "delete-model-function" && confirm("删除该模型函数？")) await api(`/api/model-functions/${target.dataset.id}`, { method: "DELETE" });
    if (action) await refresh();
  } catch (err) {
    alert(err.message);
  }
});

document.body.addEventListener("change", async (event) => {
  if (!event.target.classList.contains("upload-input")) return;
  const file = event.target.files?.[0];
  if (!file) return;
  const kind = event.target.dataset.kind;
  const id = event.target.dataset.id;
  try {
    if (kind === "pt") {
      await upload(`/api/model-functions/${id}/upload-pt`, "file", file);
      alert("PT权重文件上传并替换成功");
    } else {
      await upload(`/api/model-functions/${id}/upload-code`, "file", file);
      alert("Python逻辑代码上传并替换成功");
    }
    event.target.value = "";
    await refresh();
  } catch (err) {
    event.target.value = "";
    alert(err.message);
  }
});

document.body.addEventListener("change", async (event) => {
  if (!event.target.classList.contains("stream-model") && !event.target.classList.contains("stream-connection")) return;
  const streamId = event.target.dataset.stream;
  const stream = findStream(streamId);
  if (streamControlIsLocked(stream)) {
    render();
    return;
  }
  streamDrafts[streamId] = streamDrafts[streamId] || {};
  if (event.target.classList.contains("stream-connection")) {
    streamDrafts[streamId].connection_id = event.target.value;
    const connection = state.connections.find((item) => item.id === event.target.value);
    if (connection) {
      streamDrafts[streamId].model_id = connection.default_model_id;
      event.target.closest(".stream-controls").querySelector(".stream-model").value = connection.default_model_id;
    }
    return;
  }
  streamDrafts[streamId].model_id = event.target.value;
});

document.body.addEventListener("focusin", (event) => {
  if ((event.target.classList.contains("stream-connection") || event.target.classList.contains("stream-model")) && !event.target.disabled) {
    streamControlFocused = true;
  }
});

document.body.addEventListener("focusout", (event) => {
  if (event.target.classList.contains("stream-connection") || event.target.classList.contains("stream-model")) {
    setTimeout(() => {
      streamControlFocused = false;
      render();
    }, 150);
  }
});

if (byId("add-connection")) byId("add-connection").disabled = !canOperate;
if (byId("add-user")) byId("add-user").disabled = !canAdmin;
if (byId("add-model-function")) byId("add-model-function").disabled = !canAdmin;
if (byId("upload-model-file")) byId("upload-model-file").disabled = !canAdmin;
if (byId("add-stream")) byId("add-stream").disabled = !canAdmin;
byId("add-connection")?.addEventListener("click", () => openConnectionDialog());
byId("add-user")?.addEventListener("click", () => openUserDialog());
byId("add-model-function")?.addEventListener("click", () => openModelFunctionDialog());
byId("upload-model-file")?.addEventListener("click", () => byId("model-file-input")?.click());
byId("model-file-input")?.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    await upload("/api/model-files", "file", file);
    event.target.value = "";
    await refresh();
  } catch (err) {
    event.target.value = "";
    alert(err.message);
  }
});
byId("refresh-logs")?.addEventListener("click", refresh);

function openConnectionDialog(connection = null) {
  const form = byId("connection-form");
  form.reset();
  form.elements.connection_id.value = connection?.id || "";
  form.elements.name.value = connection?.name || "";
  form.elements.type.value = connection?.type || "rtsp";
  form.elements.source.value = connection?.source || "";
  form.elements.default_model_id.innerHTML = modelOptions(connection?.default_model_id || "person_detector");
  form.elements.default_stream_id.innerHTML = streamOptions(connection?.default_stream_id || 1);
  byId("connection-title").textContent = connection ? "编辑连接" : "新增连接";
  byId("connection-dialog").showModal();
}

const savingForms = new Set();

async function runSave(key, button, handler) {
  if (savingForms.has(key)) return;
  savingForms.add(key);
  const originalText = button?.textContent;
  if (button) {
    button.disabled = true;
    button.textContent = "保存中";
  }
  try {
    await handler();
  } catch (err) {
    alert(err.message);
  } finally {
    savingForms.delete(key);
    if (button) {
      button.disabled = false;
      button.textContent = originalText || "保存";
    }
  }
}

function wireSave(formId, buttonId, handler) {
  const form = byId(formId);
  const button = byId(buttonId);
  const onSave = (event) => {
    event.preventDefault();
    event.stopPropagation();
    runSave(formId, button, handler);
  };
  form?.addEventListener("submit", onSave);
  button?.addEventListener("click", onSave);
  button?.addEventListener("pointerdown", (event) => {
    if (event.button && event.button !== 0) return;
    onSave(event);
  });
}

async function saveConnectionForm() {
  const form = byId("connection-form");
  if (!form.reportValidity()) return;
  const payload = {
    name: form.elements.name.value,
    type: form.elements.type.value,
    source: form.elements.source.value,
    default_model_id: form.elements.default_model_id.value,
    default_stream_id: Number(form.elements.default_stream_id.value),
  };
  const id = form.elements.connection_id.value;
  await api(id ? `/api/connections/${id}` : "/api/connections", { method: id ? "PUT" : "POST", body: JSON.stringify(payload) });
  byId("connection-dialog").close();
  await refresh();
}

wireSave("connection-form", "save-connection", saveConnectionForm);

async function saveUserForm() {
  const form = byId("user-form");
  if (!form.reportValidity()) return;
  const editing = form.elements.editing.value;
  const payload = {
    role: form.elements.role.value,
    enabled: form.elements.enabled.checked,
  };
  if (form.elements.password.value) payload.password = form.elements.password.value;
  if (editing) {
    await api(`/api/users/${editing}`, { method: "PATCH", body: JSON.stringify(payload) });
  } else {
    await api("/api/users", { method: "POST", body: JSON.stringify({ ...payload, username: form.elements.username.value, password: form.elements.password.value }) });
  }
  byId("user-dialog").close();
  await refresh();
}

wireSave("user-form", "save-user", saveUserForm);

function openUserDialog(user = null) {
  const form = byId("user-form");
  form.reset();
  form.elements.editing.value = user?.username || "";
  form.elements.username.value = user?.username || "";
  form.elements.username.disabled = Boolean(user);
  form.elements.password.required = !user;
  form.elements.role.value = user?.role || "viewer";
  form.elements.enabled.checked = user ? user.enabled : true;
  byId("user-title").textContent = user ? "编辑用户" : "新增用户";
  byId("user-dialog").showModal();
}

function openModelFunctionDialog(item = null) {
  const form = byId("model-function-form");
  if (!form) return;
  const defaultConfig = {
    model_path: "",
    logic_module: "app.func.model_unhat",
    logic_function: "unhat",
    conf: 0.25,
    model_bindings: {}
  };
  form.reset();
  form.elements.editing.value = item?.id || "";
  form.elements.id.value = item?.id || "";
  form.elements.name.value = item?.name || "";
  form.elements.task.value = item?.task || "func";
  form.elements.entrypoint.value = item?.entrypoint || "app.model_functions:build_func_model";
  form.elements.description.value = item?.description || "";
  form.elements.config.value = JSON.stringify(item?.config || defaultConfig, null, 2);
  form.elements.enabled.checked = item ? item.enabled : true;
  byId("model-function-title").textContent = item ? "编辑模型逻辑" : "新增模型逻辑";
  byId("model-function-dialog").showModal();
}

async function saveModelFunctionForm() {
  const form = byId("model-function-form");
  if (!form.reportValidity()) return;
  let config;
  try {
    config = JSON.parse(form.elements.config.value);
  } catch {
    alert("配置JSON格式不正确");
    return;
  }
  const payload = {
    id: form.elements.id.value,
    name: form.elements.name.value,
    task: form.elements.task.value,
    entrypoint: form.elements.entrypoint.value,
    description: form.elements.description.value,
    config,
    enabled: form.elements.enabled.checked,
  };
  const editing = form.elements.editing.value;
  await api(editing ? `/api/model-functions/${editing}` : "/api/model-functions", { method: editing ? "PUT" : "POST", body: JSON.stringify(payload) });
  byId("model-function-dialog").close();
  await refresh();
}

wireSave("model-function-form", "save-model-function", saveModelFunctionForm);

refresh();
syncRefreshLoop();
window.addEventListener("pagehide", stopRefreshLoop);
