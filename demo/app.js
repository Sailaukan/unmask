const els = {
  apiUrl: document.querySelector("#apiUrl"),
  model: document.querySelector("#model"),
  prompt: document.querySelector("#prompt"),
  tokens: document.querySelector("#tokens"),
  steps: document.querySelector("#steps"),
  temperature: document.querySelector("#temperature"),
  cleanTail: document.querySelector("#cleanTail"),
  runBoth: document.querySelector("#runBoth"),
  runBuffered: document.querySelector("#runBuffered"),
  runStreaming: document.querySelector("#runStreaming"),
  stop: document.querySelector("#stop"),
  serverStatus: document.querySelector("#serverStatus"),
  bufferedMeta: document.querySelector("#bufferedMeta"),
  streamingMeta: document.querySelector("#streamingMeta"),
  bufferedOutput: document.querySelector("#bufferedOutput"),
  streamingOutput: document.querySelector("#streamingOutput"),
  streamingProgress: document.querySelector("#streamingProgress"),
  stepList: document.querySelector("#stepList"),
};

let activeControllers = [];
let stopRequested = false;

function requestBody(stream) {
  return {
    model: els.model.value.trim(),
    prompt: els.prompt.value,
    stream,
    options: {
      num_predict: Number(els.tokens.value),
      num_steps: Number(els.steps.value),
      temperature: Number(els.temperature.value),
      clean_tail: els.cleanTail.checked,
    },
  };
}

function endpoint() {
  return `${els.apiUrl.value.replace(/\/+$/, "")}/api/generate`;
}

function setStatus(text, state = "") {
  els.serverStatus.textContent = text;
  els.serverStatus.className = `status-pill ${state}`.trim();
}

function setBusy(isBusy) {
  els.runBoth.disabled = isBusy;
  els.runBuffered.disabled = isBusy;
  els.runStreaming.disabled = isBusy;
  els.stop.disabled = !isBusy;
}

function elapsedLabel(startedAt) {
  return `${((performance.now() - startedAt) / 1000).toFixed(2)}s`;
}

function resetBuffered() {
  els.bufferedOutput.textContent = "";
  els.bufferedMeta.textContent = "Waiting";
}

function resetStreaming() {
  els.streamingOutput.textContent = "";
  els.streamingMeta.textContent = "Waiting";
  els.streamingProgress.style.width = "0%";
  els.stepList.replaceChildren();
}

function appendStep(step, total, done) {
  if (!step || !total) return;
  const item = document.createElement("li");
  item.textContent = `${step}/${total}`;
  if (done) item.classList.add("done");
  els.stepList.appendChild(item);
  els.stepList.scrollTop = els.stepList.scrollHeight;
}

async function runBuffered() {
  resetBuffered();
  const controller = new AbortController();
  activeControllers.push(controller);
  const startedAt = performance.now();
  els.bufferedMeta.textContent = "Running";

  try {
    const response = await fetch(endpoint(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody(false)),
      signal: controller.signal,
    });

    const payload = await response.json();
    if (!response.ok || payload.error) {
      throw new Error(JSON.stringify(payload.error || payload, null, 2));
    }

    els.bufferedOutput.textContent = payload.response || "";
    els.bufferedMeta.textContent = elapsedLabel(startedAt);
  } catch (error) {
    if (error.name === "AbortError") {
      els.bufferedMeta.textContent = "Stopped";
      return;
    }
    els.bufferedOutput.textContent = error.message;
    els.bufferedMeta.textContent = "Error";
    throw error;
  } finally {
    activeControllers = activeControllers.filter((item) => item !== controller);
  }
}

async function runStreaming() {
  resetStreaming();
  const controller = new AbortController();
  activeControllers.push(controller);
  const startedAt = performance.now();
  els.streamingMeta.textContent = "Running";

  try {
    const response = await fetch(endpoint(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody(true)),
      signal: controller.signal,
    });

    if (!response.ok || !response.body) {
      const errorText = await response.text();
      throw new Error(errorText || `HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;
        const payload = JSON.parse(line);
        if (payload.error) {
          throw new Error(JSON.stringify(payload.error, null, 2));
        }

        const step = Number(payload.diffusion_step || 0);
        const total = Number(payload.diffusion_steps || 0);
        const progress = total > 0 ? Math.min(100, (step / total) * 100) : 0;

        els.streamingOutput.textContent = payload.response || "";
        els.streamingProgress.style.width = `${progress}%`;
        els.streamingMeta.textContent = payload.done
          ? elapsedLabel(startedAt)
          : `Step ${step}/${total}`;
        appendStep(step, total, Boolean(payload.done));
      }
    }
  } catch (error) {
    if (error.name === "AbortError") {
      els.streamingMeta.textContent = "Stopped";
      return;
    }
    els.streamingOutput.textContent = error.message;
    els.streamingMeta.textContent = "Error";
    throw error;
  } finally {
    activeControllers = activeControllers.filter((item) => item !== controller);
  }
}

async function runMode(mode) {
  stopRequested = false;
  setBusy(true);
  setStatus("Running", "running");

  try {
    if (mode === "both") {
      await runStreaming();
      if (!stopRequested) {
        await runBuffered();
      }
    } else if (mode === "buffered") {
      await runBuffered();
    } else {
      await runStreaming();
    }
    setStatus("Done", "running");
  } catch (error) {
    setStatus("Error", "error");
    console.error(error);
  } finally {
    setBusy(false);
  }
}

function stopActiveRequests() {
  stopRequested = true;
  for (const controller of activeControllers) {
    controller.abort();
  }
  activeControllers = [];
  setBusy(false);
  setStatus("Stopped");
}

els.runBoth.addEventListener("click", () => runMode("both"));
els.runBuffered.addEventListener("click", () => runMode("buffered"));
els.runStreaming.addEventListener("click", () => runMode("streaming"));
els.stop.addEventListener("click", stopActiveRequests);

setStatus("Ready");
