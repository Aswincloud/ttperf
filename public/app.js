/* ttperf landing page — terminal animation, ops grid, copy-to-clipboard */
(() => {
  "use strict";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const TARGET_NS = 1234567.89; // the figure shown in the README output example

  /* ---------------------------------------------------------------
     Clipboard + toast (shared by hero button, op chips, install)
  --------------------------------------------------------------- */
  const toast = document.getElementById("toast");
  let toastTimer;
  function showToast(msg) {
    if (!toast) return;
    toast.textContent = msg;
    toast.hidden = false;
    requestAnimationFrame(() => toast.classList.add("show"));
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toast.classList.remove("show");
      setTimeout(() => { toast.hidden = true; }, 220);
    }, 1600);
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Fallback for non-secure contexts (e.g. plain-http LAN preview)
      try {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        const ok = document.execCommand("copy");
        document.body.removeChild(ta);
        return ok;
      } catch {
        return false;
      }
    }
  }

  // Hero "pip install" button + install-card command buttons
  document.querySelectorAll("[data-copy]").forEach((el) => {
    el.addEventListener("click", async () => {
      const text = el.getAttribute("data-copy");
      const ok = await copyText(text);
      showToast(ok ? `Copied:  ${text}` : "Press Ctrl/Cmd+C to copy");
      el.classList.add("copied");
      setTimeout(() => el.classList.remove("copied"), 1200);
    });
  });

  /* ---------------------------------------------------------------
     Terminal: type the command, reveal output, count up the ns
  --------------------------------------------------------------- */
  const typed = document.getElementById("typed");
  const cursor = document.getElementById("cursor");
  const termOut = document.getElementById("term-out");
  const nsEl = document.getElementById("ns");
  const COMMAND = "ttperf add";

  const fmtNs = (n) =>
    n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

  function countUp() {
    if (!nsEl) return;
    if (reduceMotion) { nsEl.textContent = fmtNs(TARGET_NS); return; }
    const duration = 1100;
    const start = performance.now();
    // ease-out cubic — settles like an instrument arriving at a reading
    const ease = (t) => 1 - Math.pow(1 - t, 3);
    function frame(now) {
      const t = Math.min((now - start) / duration, 1);
      nsEl.textContent = fmtNs(TARGET_NS * ease(t));
      if (t < 1) requestAnimationFrame(frame);
      else nsEl.textContent = fmtNs(TARGET_NS);
    }
    requestAnimationFrame(frame);
  }

  function revealOutput() {
    if (!termOut) return;
    termOut.hidden = false;
    const lines = [...termOut.querySelectorAll(".ln")];
    if (reduceMotion) {
      termOut.classList.add("play");
      lines.forEach((l) => (l.style.opacity = "1"));
      countUp();
      return;
    }
    termOut.classList.add("play");
    lines.forEach((line, i) => {
      line.style.animationDelay = `${i * 90}ms`;
    });
    // start the readout right as its line lands
    const readoutIdx = lines.findIndex((l) => l.classList.contains("readout"));
    setTimeout(countUp, Math.max(readoutIdx, 0) * 90 + 120);
  }

  function typeCommand() {
    if (!typed) return;
    if (reduceMotion) {
      typed.textContent = COMMAND;
      if (cursor) cursor.style.display = "none";
      revealOutput();
      return;
    }
    let i = 0;
    (function step() {
      if (i <= COMMAND.length) {
        typed.textContent = COMMAND.slice(0, i);
        i += 1;
        setTimeout(step, 75 + Math.random() * 45);
      } else {
        setTimeout(() => {
          if (cursor) cursor.style.display = "none";
          revealOutput();
        }, 380);
      }
    })();
  }

  // Kick off the terminal once it scrolls into view (or immediately if already visible)
  const term = document.getElementById("term");
  if (term) {
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((entries, obs) => {
        entries.forEach((e) => {
          if (e.isIntersecting) { typeCommand(); obs.disconnect(); }
        });
      }, { threshold: 0.4 });
      io.observe(term);
    } else {
      typeCommand();
    }
  }

  /* ---------------------------------------------------------------
     Operations grid — loaded from ops.json (same source as the CLI)
  --------------------------------------------------------------- */
  const grid = document.getElementById("op-grid");
  const search = document.getElementById("op-search");
  const filtersEl = document.getElementById("op-filters");
  const countEl = document.getElementById("ops-count");
  const emptyEl = document.getElementById("ops-empty");

  let allOps = [];
  let activeCat = "All";
  let query = "";

  const CATEGORY_ORDER = ["All", "Unary", "Binary", "Backward", "Reduction", "Complex", "Ternary"];

  function buildFilters(ops) {
    if (!filtersEl) return;
    const counts = ops.reduce((m, o) => ((m[o.category] = (m[o.category] || 0) + 1), m), {});
    counts.All = ops.length;
    CATEGORY_ORDER.filter((c) => counts[c]).forEach((cat) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "op-filter";
      btn.setAttribute("aria-pressed", String(cat === activeCat));
      btn.dataset.cat = cat;
      btn.innerHTML = `${cat}<span class="cnt">${counts[cat]}</span>`;
      btn.addEventListener("click", () => {
        activeCat = cat;
        filtersEl.querySelectorAll(".op-filter").forEach((b) =>
          b.setAttribute("aria-pressed", String(b.dataset.cat === cat)));
        render();
      });
      filtersEl.appendChild(btn);
    });
  }

  function render() {
    if (!grid) return;
    const q = query.trim().toLowerCase();
    const matches = allOps.filter((o) => {
      const catOk = activeCat === "All" || o.category === activeCat;
      const qOk = !q || o.name.toLowerCase().includes(q);
      return catOk && qOk;
    });

    grid.innerHTML = "";
    const frag = document.createDocumentFragment();
    matches.forEach((o) => {
      const cmd = `ttperf ${o.name}`;
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "op-chip";
      chip.dataset.cat = o.category;
      chip.title = `${cmd}   ·   ${o.dtype}, ${o.layout}, ${o.shape}`;
      chip.setAttribute("aria-label", `Copy command ${cmd}`);
      chip.innerHTML =
        `<span class="op-name">${o.name}</span><span class="op-cat">${o.category}</span>`;
      chip.addEventListener("click", async () => {
        const ok = await copyText(cmd);
        showToast(ok ? `Copied:  ${cmd}` : "Press Ctrl/Cmd+C to copy");
        chip.classList.add("copied");
        setTimeout(() => chip.classList.remove("copied"), 1100);
      });
      frag.appendChild(chip);
    });
    grid.appendChild(frag);

    if (emptyEl) emptyEl.hidden = matches.length !== 0;
    if (countEl) {
      countEl.innerHTML = `Showing <b>${matches.length}</b> of ${allOps.length} operations`;
    }
  }

  if (search) {
    search.addEventListener("input", (e) => { query = e.target.value; render(); });
  }

  fetch("/ops.json")
    .then((r) => r.json())
    .then((data) => {
      allOps = (data.operations || []).slice().sort((a, b) => a.name.localeCompare(b.name));
      buildFilters(allOps);
      render();
    })
    .catch(() => {
      if (countEl) countEl.textContent = "Could not load operations list.";
    });
})();
